import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Config } from './config';

export interface Pick {
  pick_id: number; game_id: string; player_id: number; player: string | null;
  market: string; line: number; line_type: string; side: 'over' | 'under'; tier: string;
  dk_price: number; fair_prob: number; p_model: number | null; p_consensus: number | null;
  ev: number; stake: number; flagged_ts: string; tip_time: string;
  result: string | null; pnl: number | null; clv: number | null; closing_price: number | null;
  injury_news: string | null; line_now: number | null; price_now: number | null; moved: boolean;
  real_pick_id: number | null; real_price: number | null; real_stake: number | null;
  real_result: string | null; real_pnl: number | null;
}

export interface ParlayLeg {
  pick_id: number; player: string | null; market: string; line: number; side: string;
  dk_price: number; result: string | null;
}

export interface Parlay {
  parlay_id: number; tier: string; same_game: boolean; joint_prob: number; dk_price: number;
  boost_pct: number | null; ev: number; stake: number; result: string | null;
  payout_decimal: number; legs: ParlayLeg[];
}

export interface Game {
  game_id: string; tip_time: string; status: string; home: string; away: string;
  injuries: { player_id: number; name: string | null; status: string }[];
  players: { player_id: number; name: string | null }[];
}

export interface Summary {
  bets: number; voided?: number; staked_bets?: number; staked?: number; profit?: number;
  roi?: number | null; hit_rate?: number | null; breakeven?: number; avg_ev?: number;
  avg_clv?: number | null; clv_se?: number | null;
}

@Injectable({ providedIn: 'root' })
export class Api {
  private http = inject(HttpClient);
  private config = inject(Config);

  private url(path: string): string { return `${this.config.apiUrl}${path}`; }

  private params(p: Record<string, string | null | undefined>): HttpParams {
    let out = new HttpParams();
    for (const [k, v] of Object.entries(p)) if (v) out = out.set(k, v);
    return out;
  }

  slate(day: string): Observable<{ date: string; games: Game[] }> {
    return this.http.get<{ date: string; games: Game[] }>(this.url(`/slate/${day}`));
  }
  picks(day: string, mode = 'paper'): Observable<Pick[]> {
    return this.http.get<Pick[]>(this.url(`/picks/${day}`), { params: this.params({ mode }) });
  }
  parlays(day: string, mode = 'paper'): Observable<Parlay[]> {
    return this.http.get<Parlay[]>(this.url(`/parlays/${day}`), { params: this.params({ mode }) });
  }
  projection(playerId: number, day: string): Observable<any> {
    return this.http.get(this.url(`/players/${playerId}/projection`),
      { params: this.params({ day }) });
  }
  performance(f: Record<string, string | null>): Observable<any> {
    return this.http.get(this.url('/performance'), { params: this.params(f) });
  }
  bets(f: Record<string, string | null>): Observable<any> {
    return this.http.get(this.url('/bets'), { params: this.params(f) });
  }
  review(): Observable<any> { return this.http.get(this.url('/review')); }
  backtests(): Observable<any[]> { return this.http.get<any[]>(this.url('/backtests')); }
  backtest(runId: string): Observable<any> { return this.http.get(this.url(`/backtests/${runId}`)); }
  bankroll(): Observable<any[]> { return this.http.get<any[]>(this.url('/bankroll')); }
  record(f: Record<string, string | null>): Observable<any[]> {
    return this.http.get<any[]>(this.url('/record'), { params: this.params(f) });
  }
  success(f: Record<string, string | null>): Observable<any[]> {
    return this.http.get<any[]>(this.url('/success'), { params: this.params(f) });
  }
  tail(f: Record<string, string | null>): Observable<any> {
    return this.http.get(this.url('/tail'), { params: this.params(f) });
  }
  settings(): Observable<any> { return this.http.get(this.url('/settings')); }
  exportCsv(year: number, mode: string): Observable<string> {
    return this.http.get(this.url(`/export/${year}`),
      { params: this.params({ mode }), responseType: 'text' });
  }

  placePick(id: number, price: number, stake: number): Observable<any> {
    return this.http.post(this.url(`/picks/${id}/placed`), { price, stake });
  }
  placeParlay(id: number, price: number, stake: number): Observable<any> {
    return this.http.post(this.url(`/parlays/${id}/placed`), { price, stake });
  }
  boost(pct: number, maxStake: number, kind: string): Observable<any> {
    return this.http.post(this.url('/boosts'), { pct, max_stake: maxStake, kind });
  }
  manualMilestone(body: object): Observable<any> {
    return this.http.post(this.url('/prices/manual/milestone'), body);
  }
  manualSgp(pickIds: number[], price: number, boostPct = 0): Observable<any> {
    return this.http.post(this.url('/prices/manual/sgp'),
      { pick_ids: pickIds, price, boost_pct: boostPct });
  }
  switchMarket(market: string, disabled: boolean, reason: string | null): Observable<any> {
    return this.http.post(this.url(`/markets/${market}`), { disabled, reason });
  }
}
