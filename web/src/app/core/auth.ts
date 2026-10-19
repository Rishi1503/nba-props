import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient, HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { CanActivateFn, Router } from '@angular/router';
import { catchError, firstValueFrom, throwError } from 'rxjs';
import { Config } from './config';

const KEY = 'nbaprops.session';

interface Session { token: string; expires: string; }

/** The single-user login: a bearer token from POST /login, kept until it expires. */
@Injectable({ providedIn: 'root' })
export class Auth {
  private http = inject(HttpClient);
  private config = inject(Config);
  private session = signal<Session | null>(Auth.read());
  readonly loggedIn = computed(() => {
    const s = this.session();
    return !!s && new Date(s.expires).getTime() > Date.now();
  });

  token(): string | null {
    return this.loggedIn() ? this.session()!.token : null;
  }

  async login(password: string): Promise<void> {
    const s = await firstValueFrom(
      this.http.post<Session>(`${this.config.apiUrl}/login`, { password }));
    localStorage.setItem(KEY, JSON.stringify(s));
    this.session.set(s);
  }

  logout(): void {
    localStorage.removeItem(KEY);
    this.session.set(null);
  }

  private static read(): Session | null {
    try {
      return JSON.parse(localStorage.getItem(KEY) ?? 'null');
    } catch {
      return null;
    }
  }
}

/** Adds the token to API calls; a 401 ends the session and returns to the login page. */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(Auth);
  const router = inject(Router);
  const token = auth.token();
  const authed = token && !req.url.endsWith('/login')
    ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req;
  return next(authed).pipe(catchError((err: HttpErrorResponse) => {
    if (err.status === 401 && !req.url.endsWith('/login')) {
      auth.logout();
      router.navigateByUrl('/login');
    }
    return throwError(() => err);
  }));
};

export const authGuard: CanActivateFn = () => {
  const auth = inject(Auth);
  return auth.loggedIn() ? true : inject(Router).parseUrl('/login');
};
