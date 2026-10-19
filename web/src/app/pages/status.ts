import { Component, inject, signal } from '@angular/core';
import { Api } from '../core/api';
import { MARKET_LABEL, money, pct } from '../core/format';

/** Go-live checklist, weekly review, market switches, bankroll and settings. */
@Component({
  selector: 'app-status',
  template: `
    <h1>Status</h1>
    @if (review(); as r) {
      <section class="card">
        <h2>Go-live: {{ r.golive.ready ? 'READY' : 'not yet' }}</h2>
        <ul class="checklist">@for (c of r.golive.criteria; track c.name) {
          <li [class.good]="c.ok">{{ c.ok ? '✓' : '○' }} {{ c.name }}: <span class="muted">{{ c.value }}</span></li> }</ul>
      </section>
      <section class="card">
        <h2>Season (paper)</h2>
        <p class="small">{{ r.review.season.bets }} picks · CLV {{ pct(r.review.season.avg_clv, 2, true) }}
          · ROI {{ pct(r.review.season.roi, 1, true) }}</p>
        <p class="small">CLV by week:
          @for (w of r.review.weekly_clv; track w.week) { <span>{{ pct(w.avg_clv, 2, true) }} </span> }</p>
      </section>
    }
    @if (settings(); as s) {
      <section class="card">
        <h2>Markets</h2>
        @for (k of keys; track k) {
          <div class="row"><span>{{ label(k) }}</span>
            <button (click)="toggle(k)">{{ off(k) ? 'Turn on' : 'Turn off' }}</button>
            @if (off(k)) { <span class="small warn">off: {{ reason(k) }}</span> }</div>
        }
      </section>
      <section class="card small">
        <h2>Strategy</h2>
        <p>Blend {{ s.strategy.model_weight }} model · EV floor {{ pct(s.strategy.ev_min) }}
          ({{ pct(s.strategy.ev_min_model_only) }} model-only) · {{ s.strategy.kelly_fraction }} Kelly
          · {{ pct(s.strategy.max_bet_pct) }} per bet · {{ pct(s.strategy.max_daily_pct) }} per day
          · Questionable teammates: {{ s.strategy.questionable_teammates }}</p>
        <p class="muted">Edit config/strategy.yaml to change these; the website shows them read-only.</p>
      </section>
    }
    <section class="card small">
      <h2>Bankroll</h2>
      @for (b of bankroll(); track b.entry_date) {
        <div class="row"><span>{{ b.entry_date }}</span><span>{{ money(b.bankroll) }}</span>
          <span class="muted">{{ b.note ?? '' }}</span></div>
      }
      @if (!bankroll().length) { <p class="muted">No bankroll recorded yet.</p> }
    </section>
  `,
})
export class Status {
  private api = inject(Api);
  readonly keys = ['pts', 'reb', 'ast', 'fg3m', 'pra'];
  review = signal<any>(null);
  settings = signal<any>(null);
  bankroll = signal<any[]>([]);
  readonly pct = pct; readonly money = money;
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  private control(k: string): any { return (this.settings()?.markets ?? []).find((m: any) => m.market === k); }
  off(k: string): boolean { return !!this.control(k)?.disabled; }
  reason(k: string): string { return this.control(k)?.reason ?? ''; }

  constructor() { this.load(); }

  load(): void {
    this.api.review().subscribe(r => this.review.set(r));
    this.api.settings().subscribe(s => this.settings.set(s));
    this.api.bankroll().subscribe(b => this.bankroll.set(b));
  }

  toggle(k: string): void {
    const turningOff = !this.off(k);
    const reason = turningOff ? prompt(`Why turn off ${this.label(k)}?`) : null;
    if (turningOff && reason === null) return;
    this.api.switchMarket(k, turningOff, reason).subscribe(() => this.load());
  }
}
