import { Injectable } from '@angular/core';

/** Runtime settings from /config.json, so one build deploys anywhere (Stage 11). */
@Injectable({ providedIn: 'root' })
export class Config {
  apiUrl = 'http://localhost:8000';

  async load(): Promise<void> {
    try {
      const res = await fetch('config.json', { cache: 'no-store' });
      if (res.ok) {
        const body = await res.json();
        if (body.apiUrl) this.apiUrl = String(body.apiUrl).replace(/\/$/, '');
      }
    } catch {
      // keep the default (local development)
    }
  }
}
