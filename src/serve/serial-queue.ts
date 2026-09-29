/**
 * Serial task queue used by the Jev shadow tap: Laya shadows never overlap, and
 * overflow is observable (`push` returns false when the queue is full).
 */
export class SerialQueue {
  private tail: Promise<void> = Promise.resolve();
  private depth = 0;

  constructor(private readonly max: number) {}

  get pending(): number {
    return this.depth;
  }

  /** returns false when the queue is full */
  push(task: () => Promise<void>): boolean {
    if (this.depth >= this.max) return false;
    this.depth += 1;
    this.tail = this.tail
      .then(task)
      .catch(() => {
        // a task must never break the chain; task bodies handle their own errors
      })
      .finally(() => {
        this.depth -= 1;
      });
    return true;
  }

  idle(): Promise<void> {
    return this.tail;
  }
}
