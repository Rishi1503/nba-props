import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';
import { Api } from '../core/api';
import { MARKET_LABEL, american, money, pct } from '../core/format';

const MARK: Record<string, string> = { win: '✓', loss: '✗', push: 'push', void: 'void' };

/** Record: what happened to every pick and parlay leg, hit rates by week, month and year,
 *  and what tailing every pick at a flat stake would have made. */
@Component({
  selector: 'app-record',
  imports: [FormsModule, BaseChartDirective],
  template: `
    <header class="row"><h1>Record</h1>
      <select [ngModel]="mode()" (ngModelChange)="mode.set($event); loadAll()" aria-label="Mode">
        <option value="paper">paper picks</option><option value="real">real bets</option></select>
    </header>

    <section class="card">
      <h2>If you'd tailed every pick</h2>
      <form class="row" (ngSubmit)="loadTail()">
        <label>Per bet <input type="number" name="stake" min="1" step="1" [(ngModel)]="stake" /></label>
        <label>Since <input type="date" name="since" [(ngModel)]="since" /></label>
        <label class="small"><input type="checkbox" name="par" [(ngModel)]="includeParlays" /> parlays</label>
        <label class="small"><input type="checkbox" name="trk" [(ngModel)]="includeTracked" /> tracked flags</label>
        <button type="submit">Update</button>
      </form>
      @if (tail(); as t) {
        @if (t.bets) {
          <p class="big tail" [class.good]="t.profit > 0" [class.error]="t.profit < 0">
            You'd be {{ t.profit >= 0 ? 'up' : 'down' }} {{ money(abs(t.profit)) }}</p>
          <p class="small muted">{{ money(t.stake) }} on each of {{ t.bets }} bets since {{ t.first_date }}
            · {{ money(t.wagered) }} wagered · ROI {{ pct(t.roi, 1, true) }}
            · {{ t.record.win }}-{{ t.record.loss }}{{ t.record.push + t.record.void ? ', ' + (t.record.push + t.record.void) + ' refunded' : '' }}
            @if (t.pending) { · {{ t.pending }} still open }</p>
          <canvas baseChart [type]="'line'" [data]="tailChart()" [options]="moneyAxis"></canvas>
          <p class="small muted">Each bet at the price when it was flagged; your real price can differ.</p>
        } @else { <p class="muted">No settled picks yet.</p> }
      }
    </section>

    <section class="card">
      <div class="row"><h2>Success rate</h2>
        <div class="tabs">
          @for (p of periods; track p) {
            <button [class.active]="p === period()" (click)="period.set(p); loadSuccess()">{{ p }}</button>
          }
        </div></div>
      <table>
        <tr><th>{{ period() === 'week' ? 'Week of' : period() === 'month' ? 'Month' : 'Year' }}</th>
          <th>Picks</th><th>Parlays</th><th>Legs</th></tr>
        @for (r of rates(); track r.period) {
          <tr><td>{{ r.period }}</td>
            <td>{{ r.picks.won }}-{{ r.picks.lost }} <span class="muted">{{ pct(r.picks.hit_rate, 0) }}</span></td>
            <td>{{ r.parlays.won }}-{{ r.parlays.lost }} <span class="muted">{{ pct(r.parlays.hit_rate, 0) }}</span></td>
            <td>{{ r.legs.won }}-{{ r.legs.lost }} <span class="muted">{{ pct(r.legs.hit_rate, 0) }}</span></td></tr>
        }
      </table>
      @if (!rates().length) { <p class="muted">No settled picks yet.</p> }
    </section>

    <h2>Day by day</h2>
    @for (d of days(); track d.date) {
      <details class="card day" [open]="$first">
        <summary class="row"><strong>{{ d.date }}</strong>
          <span>{{ d.record.win }}-{{ d.record.loss }}{{ d.pending ? ' · ' + d.pending + ' open' : '' }}</span></summary>
        @for (p of d.picks; track p.pick_id) {
          <div class="row small pick" [class.muted]="p.stake <= 0">
            <span class="mark" [class.good]="p.result === 'win'" [class.error]="p.result === 'loss'">{{ mark(p.result) }}</span>
            <span class="grow">{{ p.player ?? p.player_id }} {{ label(p.market) }} {{ p.side }} {{ p.line }}</span>
            <span>{{ am(p.dk_price) }}{{ p.stake <= 0 ? ' · tracked' : '' }}</span>
          </div>
        }
        @for (par of d.parlays; track par.parlay_id) {
          <div class="parlay small">
            <div class="row"><strong>{{ par.tier }} parlay</strong>
              <span [class.good]="par.result === 'win'" [class.error]="par.result === 'loss'">
                {{ par.result ? (par.result === 'win' ? 'cashed' : par.result) : 'open' }} · {{ par.legs_hit }}/{{ par.legs.length }} legs</span></div>
            @for (l of par.legs; track l.pick_id) {
              <div class="row leg">
                <span class="mark" [class.good]="l.result === 'win'" [class.error]="l.result === 'loss'">{{ mark(l.result) }}</span>
                <span class="grow">{{ l.player }} {{ label(l.market) }} {{ l.side }} {{ l.line }}</span>
                <span>{{ am(l.dk_price) }}</span>
              </div>
            }
          </div>
        }
      </details>
    }
    @if (!days().length) { <p class="muted">No picks yet.</p> }
  `,
  styles: [`
    .tail { font-size: 1.6rem; margin: 10px 0 4px; }
    .pick, .leg { flex-wrap: nowrap; gap: 8px; padding: 3px 0; }
    .grow { flex: 1; }
    .mark { width: 2.4em; font-weight: 700; }
    .parlay { border-top: 1px solid var(--line); margin-top: 8px; padding-top: 6px; }
    .leg { padding-left: 8px; }
    details.day summary { list-style: none; }
  `],
})
export class RecordPage {
  private api = inject(Api);
  readonly periods = ['week', 'month', 'year'];
  mode = signal('paper');
  period = signal('week');
  stake = 10;
  since = '';
  includeParlays = true;
  includeTracked = false;
  tail = signal<any>(null);
  rates = signal<any[]>([]);
  days = signal<any[]>([]);
  readonly money = money; readonly pct = pct; readonly am = american;
  readonly moneyAxis: ChartConfiguration<'line'>['options'] = {
    plugins: { legend: { display: false } },
    scales: { y: { ticks: { callback: v => money(Number(v)) } } },
  };
  tailChart = computed<ChartConfiguration<'line'>['data']>(() => {
    const daily = this.tail()?.daily ?? [];
    return { labels: daily.map((d: any) => d.date),
             datasets: [{ data: daily.map((d: any) => d.cumulative), tension: 0.2, pointRadius: 0,
                          borderColor: '#1f6feb', borderWidth: 2, fill: true,
                          backgroundColor: 'rgba(31, 111, 235, 0.12)' }] };
  });
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  mark(r: string | null): string { return r ? MARK[r] ?? r : '·'; }
  abs(x: number): number { return Math.abs(x); }

  constructor() { this.loadAll(); }

  loadAll(): void { this.loadTail(); this.loadSuccess(); this.loadDays(); }

  loadTail(): void {
    this.api.tail({ stake: String(this.stake || 10), mode: this.mode(), start: this.since || null,
      include_parlays: String(this.includeParlays), include_tracked: String(this.includeTracked) })
      .subscribe(t => this.tail.set(t));
  }

  loadSuccess(): void {
    this.api.success({ mode: this.mode(), period: this.period() }).subscribe(r => this.rates.set(r));
  }

  loadDays(): void {
    this.api.record({ mode: this.mode() }).subscribe(d => this.days.set(d));
  }
}
