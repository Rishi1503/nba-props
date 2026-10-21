import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration } from 'chart.js';
import { Api } from '../core/api';
import { MARKET_LABEL, money, pct } from '../core/format';

/** Performance: CLV over time, calibration (predicted vs actual), ROI by stat and edge. */
@Component({
  selector: 'app-performance',
  imports: [FormsModule, BaseChartDirective],
  template: `
    <h1>Performance</h1>
    <form class="row small" (ngSubmit)="load()">
      <select name="mode" [(ngModel)]="mode"><option value="paper">paper</option>
        <option value="real">real</option></select>
      <select name="market" [(ngModel)]="market"><option value="">all stats</option>
        @for (k of markets; track k) { <option [value]="k">{{ label(k) }}</option> }</select>
      <select name="tier" [(ngModel)]="tier"><option value="">all tiers</option>
        <option value="high_conf">high-conf</option><option value="standard">standard</option></select>
      <input type="date" name="start" [(ngModel)]="start" /><input type="date" name="end" [(ngModel)]="end" />
      <button type="submit">Apply</button>
    </form>
    @if (data(); as d) {
      @if (d.summary.bets) {
        <p class="big">CLV {{ pct(d.summary.avg_clv, 2, true) }}
          <span class="small muted">± {{ pct(d.summary.clv_se, 2) }} · ROI {{ pct(d.summary.roi, 1, true) }}
          · {{ d.summary.bets }} picks · hit {{ pct(d.summary.hit_rate) }} vs breakeven {{ pct(d.summary.breakeven) }}
          · P&L {{ money(d.summary.profit) }}</span></p>
        @if (d.summary.bets < 300) { <p class="small warn">Under 300 picks: still noisy. Judge by CLV.</p> }
        <h2>CLV by day</h2><canvas baseChart [type]="'line'" [data]="clvChart()" [options]="pctAxis"></canvas>
        <h2>Calibration</h2>
        <p class="small muted">Predicted win probability vs how often picks won; on the diagonal is calibrated.</p>
        <canvas baseChart [type]="'scatter'" [data]="calChart()" [options]="calOptions"></canvas>
        <h2>By stat</h2>
        <table><tr><th>Stat</th><th>Picks</th><th>CLV</th><th>ROI</th><th>Hit</th></tr>
          @for (row of byMarket(); track row.k) {
            <tr><td>{{ label(row.k) }}</td><td>{{ row.v.bets }}</td><td>{{ pct(row.v.avg_clv, 2, true) }}</td>
              <td>{{ pct(row.v.roi, 1, true) }}</td><td>{{ pct(row.v.hit_rate) }}</td></tr>
          }</table>
        <h2>By edge</h2>
        <table><tr><th>EV</th><th>Picks</th><th>Predicted EV</th><th>Realized CLV</th></tr>
          @for (row of byEv(); track row.k) {
            <tr><td>{{ row.k }}</td><td>{{ row.v.bets }}</td><td>{{ pct(row.v.predicted_ev, 1, true) }}</td>
              <td>{{ pct(row.v.avg_clv, 2, true) }}</td></tr>
          }</table>
      } @else { <p class="muted">No settled picks for these filters yet.</p> }
    }
  `,
})
export class Performance {
  private api = inject(Api);
  readonly markets = ['pts', 'reb', 'ast', 'fg3m', 'pra'];
  mode = 'paper'; market = ''; tier = ''; start = ''; end = '';
  data = signal<any>(null);
  readonly pct = pct; readonly money = money;
  readonly pctAxis: ChartConfiguration<'line'>['options'] = {
    plugins: { legend: { display: false } },
    scales: { y: { ticks: { callback: v => `${(Number(v) * 100).toFixed(1)}%` } } },
  };
  readonly calOptions: ChartConfiguration<'scatter'>['options'] = {
    plugins: { legend: { display: false } },
    scales: { x: { min: 0, max: 1, title: { display: true, text: 'predicted' } },
              y: { min: 0, max: 1, title: { display: true, text: 'actual' } } },
  };
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  byMarket = computed(() => Object.entries(this.data()?.by_market ?? {}).map(([k, v]) => ({ k, v: v as any })));
  byEv = computed(() => Object.entries(this.data()?.by_ev ?? {}).map(([k, v]) => ({ k, v: v as any })));
  clvChart = computed<ChartConfiguration<'line'>['data']>(() => {
    const daily = this.data()?.daily ?? [];
    return { labels: daily.map((d: any) => d.game_date),
             datasets: [{ data: daily.map((d: any) => d.clv), tension: 0.2, pointRadius: 2,
                          borderColor: '#1f6feb', backgroundColor: '#1f6feb', borderWidth: 2 }] };
  });
  calChart = computed<ChartConfiguration<'scatter'>['data']>(() => {
    const bins = this.data()?.calibration ?? [];
    return { datasets: [
      { data: bins.map((b: any) => ({ x: b.predicted, y: b.observed })), pointRadius: 5,
        backgroundColor: '#1f6feb', borderColor: '#1f6feb' },
      { data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderDash: [4, 4],
        borderColor: '#9aa0a6', borderWidth: 1.5 },
    ] };
  });

  constructor() { this.load(); }

  load(): void {
    this.api.performance({ mode: this.mode, market: this.market || null, tier: this.tier || null,
      start: this.start || null, end: this.end || null }).subscribe(d => this.data.set(d));
  }
}
