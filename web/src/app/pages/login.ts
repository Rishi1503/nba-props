import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Auth } from '../core/auth';

@Component({
  selector: 'app-login',
  imports: [FormsModule],
  template: `
    <section class="card narrow">
      <h1>NBA props</h1>
      <form (ngSubmit)="submit()">
        <label>Password <input type="password" name="pw" [(ngModel)]="password"
          autocomplete="current-password" required autofocus /></label>
        <button type="submit" [disabled]="busy()">Log in</button>
      </form>
      @if (error()) { <p class="error">{{ error() }}</p> }
    </section>
  `,
})
export class Login {
  private auth = inject(Auth);
  private router = inject(Router);
  password = '';
  busy = signal(false);
  error = signal<string | null>(null);

  async submit(): Promise<void> {
    this.busy.set(true);
    this.error.set(null);
    try {
      await this.auth.login(this.password);
      this.router.navigateByUrl('/');
    } catch (e: any) {
      this.error.set(e?.status === 429 ? 'Too many attempts: wait 15 minutes.'
        : e?.status === 401 ? 'Wrong password.' : 'Could not reach the API.');
    } finally {
      this.busy.set(false);
    }
  }
}
