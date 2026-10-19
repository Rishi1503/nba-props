import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Api } from '../core/api';
import { MARKET_LABEL, american, money, pct } from '../core/format';

/** Bet tracker: placed bets, graded each morning, with running totals; the tax-year CSV. */
@Component({
  selector: 'app-tracker',
  imports: [FormsModule],
  template: `
    <header class="row"><h1>Bets</h1>
      <select [ngModel]="mode()" (ngModelChange)="mode.set($event); load()">
        <option value="real">real money</option><option value="paper">paper</option></select>
    </header>
    @if (data(); as d) {
      <p class="big">{{ money(d.totals.pnl) }} <span class="muted small">on {{ money(d.totals.staked) }}
        · {{ d.totals.record?.win ?? 0 }}-{{ d.totals.record?.loss ?? 0 }}
        · {{ d.totals.open }} open</span></p>
      @for (b of reversed(); track b.kind + b.id) {
        <article class="card small">
          <div class="row"><span>{{ b.date }} · {{ b.player ?? b.kind }}
            @if (b.market) { {{ label(b.market) }} {{ b.side }} {{ b.line }} }</span>
            <span class="price">{{ am(b.odds) }}</span></div>
          <div class="row"><span>stake {{ money(b.stake) }}</span>
            <span [class.good]="b.pnl > 0" [class.error]="b.pnl < 0">{{ b.result ?? 'open' }}
              {{ b.result ? money(b.pnl) : '' }}</span>
            @if (b.clv !== null) { <span>CLV {{ pct(b.clv, 1, true) }}</span> }
            <span class="muted">running {{ money(b.running_pnl) }}</span></div>
        </article>
      }
      @if (!d.bets.length) { <p class="muted">No bets yet.</p> }
    }
    <section class="card">
      <h2>Tax records</h2>
      <p class="small muted">Every settled bet for the year (date, market, stake, odds, result).</p>
      <div class="row"><input type="number" [(ngModel)]="year" aria-label="Year" />
        <button (click)="download()">Download CSV</button></div>
    </section>
  `,
})
export class Tracker {
  private api = inject(Api);
  mode = signal('real');
  data = signal<any>(null);
  year = new Date().getFullYear();
  readonly am = american; readonly money = money; readonly pct = pct;
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  reversed(): any[] { return [...(this.data()?.bets ?? [])].reverse(); }

  constructor() { this.load(); }

  load(): void { this.api.bets({ mode: this.mode() }).subscribe(d => this.data.set(d)); }

  download(): void {
    this.api.exportCsv(this.year, this.mode()).subscribe(csv => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      a.download = `bets-${this.year}-${this.mode()}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }
}
