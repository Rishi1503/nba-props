import { Routes } from '@angular/router';
import { authGuard } from './core/auth';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./pages/login').then(m => m.Login) },
  { path: '', canActivate: [authGuard], loadComponent: () => import('./pages/board').then(m => m.Board) },
  { path: 'player/:id', canActivate: [authGuard], loadComponent: () => import('./pages/player').then(m => m.Player) },
  { path: 'bets', canActivate: [authGuard], loadComponent: () => import('./pages/tracker').then(m => m.Tracker) },
  { path: 'performance', canActivate: [authGuard], loadComponent: () => import('./pages/performance').then(m => m.Performance) },
  { path: 'backtests', canActivate: [authGuard], loadComponent: () => import('./pages/backtests').then(m => m.Backtests) },
  { path: 'status', canActivate: [authGuard], loadComponent: () => import('./pages/status').then(m => m.Status) },
  { path: '**', redirectTo: '' },
];
