import { Component, computed, inject, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';
import { Api } from '../core/api';
import { MARKET_LABEL, american, timeNY, todayNY } from '../core/format';

/** Player detail: simulated distribution with DraftKings' line, last 10 games, line moves. */
@Component({
  selector: 'app-player',
  imports: [BaseChartDirective],
  template: `
    @if (data(); as d) {
      <h1>{{ d.name }}</h1>
      <div class="tabs">
        @for (k of marketKeys(); track k) {
          <button [class.active]="k === market()" (click)="market.set(k)">{{ label(k) }}</button>
        }
      </div>
      @if (current(); as c) {
        <p class="small">mean {{ c.mean.toFixed(1) }} · sd {{ c.sd.toFixed(1) }}
          @if (line() !== null) { · DraftKings line {{ line() }} }
          @if (c.low_confidence) { · <span class="badge warn">low confidence</span> }
          @if (c.questionable_scenarios) { · <span class="badge">Questionable teammate</span> }</p>
        <canvas baseChart [type]="'bar'" [data]="chart()" [options]="options"></canvas>
      } @else { <p class="muted">No projection for this date.</p> }
      <h2>Last 10 games</h2>
      <table><tr><th>Date</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th><th>3PM</th></tr>
        @for (g of d.recent_games; track g.game_date) {
          <tr><td>{{ g.game_date }}</td><td>{{ g.dnp ? 'DNP' : (g.minutes ?? 0).toFixed(0) }}</td>
            <td>{{ g.pts }}</td><td>{{ g.reb }}</td><td>{{ g.ast }}</td><td>{{ g.fg3m }}</td></tr>
        }</table>
      <h2>DraftKings line today</h2>
      <table><tr><th>Time</th><th>Line</th><th>Over</th><th>Under</th></tr>
        @for (l of moves(); track l.snapshot_ts) {
          <tr><td>{{ time(l.snapshot_ts) }}</td>
            <td>{{ l.is_withdrawn ? 'pulled' : l.line }}</td>
            <td>{{ am(l.over_price) }}</td><td>{{ am(l.under_price) }}</td></tr>
        }</table>
    } @else { <p class="muted">Loading…</p> }
  `,
})
export class Player {
  private api = inject(Api);
  private route = inject(ActivatedRoute);
  data = signal<any>(null);
  market = signal('pts');
  readonly am = american;
  readonly time = timeNY;
  readonly options: ChartConfiguration<'bar'>['options'] = {
    responsive: true, plugins: { legend: { display: false } },
    scales: { y: { ticks: { callback: v => `${Math.round(Number(v) * 100)}%` } } },
  };
  marketKeys = computed(() => Object.keys(this.data()?.markets ?? {}));
  current = computed(() => this.data()?.markets?.[this.market()]);
  moves = computed(() => (this.data()?.line_history ?? []).filter((l: any) => l.market === this.market()));
  line = computed(() => {
    const live = this.moves().filter((l: any) => !l.is_withdrawn);
    return live.length ? live[live.length - 1].line : null;
  });
  chart = computed<ChartConfiguration<'bar'>['data']>(() => {
    // trim the near-empty tails so the axis covers the plausible range
    const all: any[] = this.current()?.histogram ?? [];
    const lo = all.findIndex(h => h.p >= 0.001);
    const hi = all.length - [...all].reverse().findIndex(h => h.p >= 0.001);
    const hist = lo >= 0 ? all.slice(lo, hi) : all;
    const line = this.line();
    return {
      labels: hist.map((h: any) => h.value),
      datasets: [{
        data: hist.map((h: any) => h.p),
        // over the line in one color, under in another: the shaded area is P(over)
        backgroundColor: hist.map((h: any) => line !== null && h.value > line ? '#2f855a' : '#a0aec0'),
      }],
    };
  });
  label(k: string): string { return MARKET_LABEL[k] ?? k; }

  constructor() {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    const day = this.route.snapshot.queryParamMap.get('day') ?? todayNY();
    this.api.projection(id, day).subscribe(d => {
      this.data.set(d);
      if (!d.markets[this.market()] && Object.keys(d.markets).length) {
        this.market.set(Object.keys(d.markets)[0]);
      }
    });
  }
}
