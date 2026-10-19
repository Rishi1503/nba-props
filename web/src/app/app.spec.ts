import { describe, expect, it } from 'vitest';
import { routes } from './app.routes';

describe('routes', () => {
  it('guards every page except login', () => {
    const pages = routes.filter(r => r.path !== '**');
    for (const r of pages) {
      if (r.path === 'login') expect(r.canActivate).toBeUndefined();
      else expect(r.canActivate?.length).toBe(1);
    }
  });
});
