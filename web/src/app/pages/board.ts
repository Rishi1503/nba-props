import { NgTemplateOutlet } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';
import { Api, Game, Parlay, Pick } from '../core/api';
import { MARKET_LABEL, american, fairAmerican, money, pct, timeNY, todayNY } from '../core/format';

/** Today's board (plan, Stage 10): picks by EV, the parlay panel, boosts, manual prices. */
@Component({
  selector: 'app-board',
  imports: [FormsModule, NgTemplateOutlet, RouterLink],
  template: `
    <header class="row">
      <h1>Board</h1>
      <input type="date" [ngModel]="day()" (ngModelChange)="load($event)" aria-label="Date" />
    </header>
    @if (error()) { <p class="error">{{ error() }}</p> }
    @if (loading()) { <p class="muted">Loading…</p> }

    <section>
      <h2>{{ staked().length }} to bet <span class="muted">· {{ tracked().length }} tracked</span></h2>
      @for (p of staked(); track p.pick_id) { <ng-container *ngTemplateOutlet="card; context: { $implicit: p }" /> }
      @if (!loading() && !staked().length) { <p class="muted">No staked picks for this date.</p> }
      @if (tracked().length) {
        <details>
          <summary>Tracked flags (not staked)</summary>
          @for (p of tracked(); track p.pick_id) { <ng-container *ngTemplateOutlet="card; context: { $implicit: p }" /> }
        </details>
      }
    </section>

    <ng-template #card let-p>
      <article class="card" [class.hc]="p.tier === 'high_conf'">
        <div class="row">
          <a [routerLink]="['/player', p.player_id]" [queryParams]="{ day: day() }">
            <strong>{{ p.player ?? p.player_id }}</strong></a>
          <span class="muted">{{ time(p.tip_time) }}</span>
        </div>
        <div class="big">{{ label(p.market) }} {{ p.side }} {{ p.line }}
          <span class="price">{{ am(p.dk_price) }}</span></div>
        <div class="row small">
          <span>fair {{ fair(p.fair_prob) }} ({{ pc(p.fair_prob) }})</span>
          <span>EV {{ pc(p.ev, 1, true) }}</span>
          <span>stake {{ m(p.stake) }}</span>
        </div>
        <div class="badges">
          @if (p.tier === 'high_conf') { <span class="badge good">high-conf</span> }
          @if (p.p_consensus === null) { <span class="badge">model only</span> }
          @if (p.injury_news) { <span class="badge warn" [title]="p.injury_news">injury news</span> }
          @if (p.moved) { <span class="badge warn">now {{ p.line_now }} {{ am(p.price_now) }}</span> }
          @if (p.result) { <span class="badge">{{ p.result }} {{ m(p.pnl) }}</span> }
        </div>
        @if (p.real_pick_id) {
          <p class="small good">Placed: {{ m(p.real_stake) }} at {{ am(p.real_price) }}
            @if (p.real_result) { · {{ p.real_result }} {{ m(p.real_pnl) }} }</p>
        } @else if (placing() === p.pick_id) {
          <form class="row" (ngSubmit)="place(p)">
            <label>Price <input type="number" name="price" [(ngModel)]="placePrice" required /></label>
            <label>Stake <input type="number" name="stake" step="0.01" [(ngModel)]="placeStake" required /></label>
            <button type="submit">Save</button>
            <button type="button" class="link" (click)="placing.set(null)">Cancel</button>
          </form>
        } @else {
          <div class="row">
            <button (click)="startPlacing(p)">Placed</button>
            <label class="small"><input type="checkbox" [checked]="sgp().has(p.pick_id)"
              (change)="toggleSgp(p.pick_id)" /> same-game parlay leg</label>
          </div>
        }
      </article>
    </ng-template>

    <section>
      <h2>Parlays</h2>
      @for (par of parlays(); track par.parlay_id) {
        <article class="card">
          <div class="row"><strong>{{ par.tier }}{{ par.same_game ? ' (same game)' : '' }}</strong>
            <span class="price">{{ am(par.dk_price) }}{{ par.boost_pct ? ' +' + pc(par.boost_pct, 0) + ' boost' : '' }}</span></div>
          <ul>@for (l of par.legs; track l.pick_id) {
            <li>{{ l.player }} {{ label(l.market) }} {{ l.side }} {{ l.line }} ({{ am(l.dk_price) }})</li> }</ul>
          <div class="row small"><span>all hit {{ pc(par.joint_prob) }}</span>
            <span>EV {{ pc(par.ev, 1, true) }}</span><span>stake {{ m(par.stake) }}</span></div>
          <button (click)="placeParlay(par)">Placed</button>
        </article>
      }
      @if (!parlays().length) { <p class="muted">No parlays for this date.</p> }
    </section>

    <section class="card">
      <h2>Boost</h2>
      <form class="row" (ngSubmit)="boost()">
        <label>Boost % <input type="number" name="bp" [(ngModel)]="boostPct" min="1" required /></label>
        <label>Max stake <input type="number" name="bm" [(ngModel)]="boostMax" required /></label>
        <select name="bk" [(ngModel)]="boostKind"><option value="parlay">parlay</option>
          <option value="single">single bet</option></select>
        <button type="submit">Price it</button>
      </form>
      @if (boostResult()) { <pre class="small">{{ boostResult() }}</pre> }
    </section>

    <section class="card">
      <h2>Manual price</h2>
      <p class="small muted">Same-game parlay: tick two or more picks from one game above
        ({{ sgp().size }} ticked), then enter DraftKings' price.</p>
      <form class="row" (ngSubmit)="priceSgp()">
        <label>SGP price <input type="number" name="sp" [(ngModel)]="sgpPrice" /></label>
        <button type="submit" [disabled]="sgp().size < 2">Price SGP</button>
      </form>
      <p class="small muted">Milestone missing from the feed (e.g. "20+ points" is line 19.5):</p>
      <form class="row" (ngSubmit)="priceMilestone()">
        <select name="mg" [(ngModel)]="msGame" (ngModelChange)="pickGame($event)">
          @for (g of games(); track g.game_id) { <option [value]="g.game_id">{{ g.away }} at {{ g.home }}</option> }
        </select>
        <select name="mp" [(ngModel)]="msPlayer" aria-label="Player">
          @for (pl of playersOf(msGame); track pl.player_id) {
            <option [ngValue]="pl.player_id">{{ pl.name ?? pl.player_id }}</option> }
        </select>
        <select name="mm" [(ngModel)]="msMarket">
          @for (k of markets; track k) { <option [value]="k">{{ label(k) }}</option> }</select>
        <label>Line <input type="number" step="0.5" name="ml" [(ngModel)]="msLine" /></label>
        <label>Price <input type="number" name="mpr" [(ngModel)]="msPrice" /></label>
        <button type="submit">Price</button>
      </form>
      @if (manualResult()) { <pre class="small">{{ manualResult() }}</pre> }
    </section>
  `,
})
export class Board {
  private api = inject(Api);
  readonly markets = ['pts', 'reb', 'ast', 'fg3m', 'pra'];
  day = signal(todayNY());
  loading = signal(false);
  error = signal<string | null>(null);
  picks = signal<Pick[]>([]);
  parlays = signal<Parlay[]>([]);
  games = signal<Game[]>([]);
  staked = computed(() => this.picks().filter(p => p.stake > 0));
  tracked = computed(() => this.picks().filter(p => p.stake <= 0));
  placing = signal<number | null>(null);
  sgp = signal(new Set<number>());
  boostResult = signal<string | null>(null);
  manualResult = signal<string | null>(null);
  placePrice = 0; placeStake = 0;
  boostPct = 30; boostMax = 25; boostKind = 'parlay';
  sgpPrice = 0;
  msGame = ''; msPlayer = 0; msMarket = 'pts'; msLine = 19.5; msPrice = 150;

  readonly am = american; readonly pc = pct; readonly m = money; readonly fair = fairAmerican;
  readonly time = timeNY;
  label(k: string): string { return MARKET_LABEL[k] ?? k; }
  playersOf(gameId: string) { return this.games().find(g => g.game_id === gameId)?.players ?? []; }

  /** Select a game and its first player, so the form's value matches what's displayed. */
  pickGame(gameId: string): void {
    this.msGame = gameId;
    this.msPlayer = this.playersOf(gameId)[0]?.player_id ?? 0;
  }

  constructor() { this.load(this.day()); }

  load(day: string): void {
    this.day.set(day);
    this.loading.set(true);
    this.error.set(null);
    forkJoin({ picks: this.api.picks(day), parlays: this.api.parlays(day), slate: this.api.slate(day) })
      .subscribe({
        next: ({ picks, parlays, slate }) => {
          this.picks.set(picks);
          this.parlays.set(parlays);
          this.games.set(slate.games);
          if (slate.games.length && !this.playersOf(this.msGame).length) {
            this.pickGame(slate.games[0].game_id);
          }
          this.loading.set(false);
        },
        error: e => { this.error.set(e.error?.detail ?? 'Could not load the board.'); this.loading.set(false); },
      });
  }

  startPlacing(p: Pick): void {
    this.placing.set(p.pick_id);
    this.placePrice = p.price_now ?? p.dk_price;
    this.placeStake = p.stake;
  }

  place(p: Pick): void {
    this.api.placePick(p.pick_id, this.placePrice, this.placeStake).subscribe({
      next: () => { this.placing.set(null); this.load(this.day()); },
      error: e => this.error.set(e.error?.detail ?? 'Could not save the bet.'),
    });
  }

  placeParlay(par: Parlay): void {
    const price = Number(prompt('Price you got (American odds)', String(par.dk_price)));
    const stake = Number(prompt('Stake', String(par.stake)));
    if (!price || !stake) return;
    this.api.placeParlay(par.parlay_id, price, stake).subscribe({
      next: () => this.load(this.day()),
      error: e => this.error.set(e.error?.detail ?? 'Could not save the parlay.'),
    });
  }

  toggleSgp(id: number): void {
    const next = new Set(this.sgp());
    next.has(id) ? next.delete(id) : next.add(id);
    this.sgp.set(next);
  }

  boost(): void {
    this.api.boost(this.boostPct / 100, this.boostMax, this.boostKind).subscribe({
      next: r => {
        this.boostResult.set(r.boosted.length
          ? r.boosted.map((b: any) => `${b.legs.map((l: any) => `${l.player_id} ${l.market} ${l.side} ${l.line}`).join(' + ')}: `
            + `${american(b.price)} boosted, EV ${pct(b.ev, 1, true)}, stake ${money(b.stake)}`).join('\n')
          : 'Nothing qualifies for this boost.');
        this.load(this.day());
      },
      error: e => this.boostResult.set(e.error?.detail ?? 'Boost failed.'),
    });
  }

  priceSgp(): void {
    this.api.manualSgp([...this.sgp()], this.sgpPrice).subscribe({
      next: r => this.manualResult.set(`All legs hit ${pct(r.joint_prob)} · EV ${pct(r.ev, 1, true)} · `
        + `stake ${money(r.stake)}${r.flag ? '' : ' (not +EV)'}\n${(r.notes ?? []).join(' ')}`),
      error: e => this.manualResult.set(e.error?.detail ?? 'Could not price it.'),
    });
  }

  priceMilestone(): void {
    this.api.manualMilestone({ game_id: this.msGame, player_id: this.msPlayer, market: this.msMarket,
      line: this.msLine, price: this.msPrice }).subscribe({
      next: r => this.manualResult.set(r.error ?? `Fair ${pct(r.p_fair)} · EV ${pct(r.ev, 1, true)} · `
        + `stake ${money(r.stake)}${r.flag ? '' : ' (no bet)'}`),
      error: e => this.manualResult.set(e.error?.detail ?? 'Could not price it.'),
    });
  }
}
