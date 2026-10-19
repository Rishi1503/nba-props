import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { Api } from '../core/api';
import { MARKET_LABEL, pct } from '../core/format';

/** Backtests: two runs side by side before promoting a model (plan, Stage 10). */
@Component({
  selector: 'app-backtests',
  imports: [FormsModule],
  template: `
    <h1>Backtests</h1>
    @if (runs().length) {
      <div class="row">
        <select [(ngModel)]="a">@for (r of runs(); track r.run_id) { <option [value]="r.run_id">{{ r.run_id }}</option> }</select>
        <span>vs</span>
        <select [(ngModel)]="b">@for (r of runs(); track r.run_id) { <option [value]="r.run_id">{{ r.run_id }}</option> }</select>
        <button (click)="compare()">Compare</button>
      </div>
    } @else { <p class="muted">No backtests stored yet (make backtest-tune).</p> }
    @if (pair(); as p) {
      <p class="small">{{ a }}: {{ p[0].holdout }} · targets {{ p[0].layer_a.targets.passes ? 'PASS' : 'FAIL' }}<br />
        {{ b }}: {{ p[1].holdout }} · targets {{ p[1].layer_a.targets.passes ? 'PASS' : 'FAIL' }}</p>
      <table>
        <tr><th>Stat</th><th>MAE vs baseline</th><th>Log loss</th><th>ECE</th><th>PIT</th></tr>
        @for (k of markets(p); track k) {
          @for (side of [0, 1]; track side) {
            <tr [class.muted]="side === 1"><td>{{ side === 0 ? label(k) : '' }} {{ side === 0 ? a : b }}</td>
              <td>{{ pct(p[side].layer_a.markets[k].accuracy.mae_improvement, 1, true) }}</td>
              <td>{{ p[side].layer_a.markets[k].probabilities.log_loss_calibrated.toFixed(4) }}</td>
              <td>{{ pct(p[side].layer_a.markets[k].probabilities.ece_calibrated) }}</td>
              <td>{{ p[side].layer_a.markets[k].pit.verdict }}</td></tr>
          }
        }
      </table>
    }
  `,
})
export class Backtests {
  private api = inject(Api);
  runs = signal<any[]>([]);
  pair = signal<any[] | null>(null);
  a = ''; b = '';
  readonly pct = pct;
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  markets(p: any[]): string[] { return Object.keys(p[0].layer_a.markets); }

  constructor() {
    this.api.backtests().subscribe(r => {
      this.runs.set(r);
      this.a = r[0]?.run_id ?? '';
      this.b = r[1]?.run_id ?? this.a;
    });
  }

  compare(): void {
    forkJoin([this.api.backtest(this.a), this.api.backtest(this.b)]).subscribe(p => this.pair.set(p));
  }
}
