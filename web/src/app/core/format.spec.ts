import { describe, expect, it } from 'vitest';
import { american, decimalFromAmerican, fairAmerican, money, pct, todayNY } from './format';

describe('format', () => {
  it('formats American odds', () => {
    expect(american(120)).toBe('+120');
    expect(american(-110)).toBe('-110');
    expect(american(null)).toBe('–');
  });

  it('converts American to decimal', () => {
    expect(decimalFromAmerican(100)).toBeCloseTo(2.0);
    expect(decimalFromAmerican(-110)).toBeCloseTo(1.9091, 4);
  });

  it('prices a probability fairly', () => {
    expect(fairAmerican(0.5)).toBe('+100');
    expect(fairAmerican(0.6)).toBe('-150');
    expect(fairAmerican(0.25)).toBe('+300');
    expect(fairAmerican(1)).toBe('–');
  });

  it('formats percents and money', () => {
    expect(pct(0.0312, 1, true)).toBe('+3.1%');
    expect(pct(-0.02, 1, true)).toBe('-2.0%');
    expect(money(-12.5)).toBe('-$12.50');
  });

  it("uses New York's date, not the device's", () => {
    // 03:30 UTC on Oct 21 is still Oct 20 in New York (EDT)
    expect(todayNY(new Date('2026-10-21T03:30:00Z'))).toBe('2026-10-20');
  });
});
