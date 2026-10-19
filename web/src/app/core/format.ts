/** Odds and number formatting shared by every page. */

export function american(price: number | null | undefined): string {
  if (price === null || price === undefined) return '–';
  return price > 0 ? `+${price}` : `${price}`;
}

export function decimalFromAmerican(price: number): number {
  return price > 0 ? 1 + price / 100 : 1 + 100 / Math.abs(price);
}

/** The American price that is fair for a win probability (no vig). */
export function fairAmerican(p: number | null | undefined): string {
  if (p === null || p === undefined || p <= 0 || p >= 1) return '–';
  const d = 1 / p;
  return american(d >= 2 ? Math.round((d - 1) * 100) : Math.round(-100 / (d - 1)));
}

export function pct(x: number | null | undefined, digits = 1, signed = false): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '–';
  const s = (x * 100).toFixed(digits);
  return `${signed && x > 0 ? '+' : ''}${s}%`;
}

export function money(x: number | null | undefined): string {
  if (x === null || x === undefined) return '–';
  const sign = x < 0 ? '-' : '';
  return `${sign}$${Math.abs(x).toFixed(2)}`;
}

/** Today's date in New York (the slate's calendar), as YYYY-MM-DD. */
export function todayNY(now: Date = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(now);
  return parts; // en-CA formats as YYYY-MM-DD
}

export function timeNY(iso: string | null | undefined): string {
  if (!iso) return '';
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit',
  }).format(new Date(iso));
}

export const MARKET_LABEL: Record<string, string> = {
  pts: 'Points', reb: 'Rebounds', ast: 'Assists', fg3m: 'Threes', pra: 'Pts+Reb+Ast',
};
