#!/usr/bin/env node
/**
 * Emit ONNX-Runtime calibration tensors for the Laya bundle, built only from the
 * repo's test fixtures (scripts/fixtures.ts).
 *
 * The Laya ONNX graph has five inputs — input_ids [B,L], attention_mask [B,L],
 * marker_pos [B,K], marker_mask [B,K], qtype [B] — so ModelOpt's ONNX PTQ needs a
 * dict of numpy arrays keyed by those exact names. This script reproduces the
 * checkpoint's `build_sequence` layout with the bundle's own tokenizer and
 * writes a JSON file that scripts/quantize-laya.py turns into numpy arrays.
 *
 * It needs only laya_config.json + tokenizer/, so it runs offline against the
 * checked-in `onnx/` fixture (no 1.6 GB weights):
 *
 *   ./node_modules/.bin/tsx scripts/calibration-fixtures.ts --model-dir onnx --out /tmp/calib.json
 */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildSequence, QTYPES, toInternal, type SpecialIds } from "../src/sequence.js";
import type { LayaConfig } from "../src/types.js";
import { FIXTURES } from "./fixtures.js";

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

interface Sample {
  name: string;
  input_ids: number[];
  attention_mask: number[];
  marker_pos: number[];
  marker_mask: number[];
  qtype: number;
}

async function main(): Promise<void> {
  const modelDir = path.resolve(arg("--model-dir") ?? process.env.LAYA_SERVE_MODEL_DIR ?? "onnx");
  const outPath = path.resolve(arg("--out") ?? "scratch/laya-calibration.json");
  const read = async (file: string): Promise<unknown> => JSON.parse(await readFile(path.join(modelDir, file), "utf8"));
  const config = (await read("laya_config.json")) as LayaConfig;
  const tok = new Tokenizer((await read("tokenizer/tokenizer.json")) as object, (await read("tokenizer/tokenizer_config.json")) as object);
  const idOf = (token: string): number => {
    const id = tok.token_to_id(token);
    if (id === undefined) throw new Error(`special token ${token} missing from tokenizer`);
    return id;
  };
  const ids: SpecialIds = { cls: idOf("[CLS]"), sep: idOf("[SEP]"), mask: idOf("[MASK]"), pad: idOf("[PAD]"), maskTok: "[MASK]" };
  const encode = (text: string): number[] => tok.encode(text, { add_special_tokens: false }).ids;

  const samples: Sample[] = [];
  for (const fixture of FIXTURES) {
    for (const question of Object.values(fixture.questions)) {
      const internal = toInternal(question);
      const built = buildSequence(encode, ids, fixture.state, internal, config.max_len, config.head_max_len);
      if (built.markers.length === 0) throw new Error(`fixture ${fixture.name}: no markers produced`);
      samples.push({
        name: `${fixture.name}:${internal.t}`,
        input_ids: built.ids,
        attention_mask: built.ids.map(() => 1),
        marker_pos: built.markers,
        marker_mask: built.markers.map(() => 1),
        qtype: QTYPES[internal.t],
      });
    }
  }

  // Right-pad to one common L and K so `calibration_data` is rectangular.
  const maxL = Math.max(...samples.map((s) => s.input_ids.length));
  const maxK = Math.max(...samples.map((s) => s.marker_pos.length));
  const pad = (values: number[], length: number, fill: number): number[] => [...values, ...Array<number>(length - values.length).fill(fill)];
  const payload = {
    schema: 1,
    model_dir: modelDir,
    max_len: config.max_len,
    head_max_len: config.head_max_len,
    input_names: ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
    input_ids: samples.map((s) => pad(s.input_ids, maxL, ids.pad)),
    attention_mask: samples.map((s) => pad(s.attention_mask, maxL, 0)),
    marker_pos: samples.map((s) => pad(s.marker_pos, maxK, 0)),
    marker_mask: samples.map((s) => pad(s.marker_mask, maxK, 0)),
    qtype: samples.map((s) => s.qtype),
    samples: samples.map((s) => s.name),
  };
  await writeFile(outPath, `${JSON.stringify(payload)}\n`, "utf8");
  process.stdout.write(`calibration-fixtures: wrote ${samples.length} samples (L=${maxL}, K=${maxK}) to ${outPath}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`calibration-fixtures: fatal: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
