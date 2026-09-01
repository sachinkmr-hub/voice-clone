# VoiceGuard operations dashboard

The fleet view: every call currently under analysis, ranked by risk, updating twice a
second from the `/v1/dashboard` WebSocket.

This is a **different screen from `/console`**, on purpose:

| | `/console` (served by the API) | this dashboard |
|---|---|---|
| Question it answers | "Is *this* call real?" | "Which of my 200 live calls needs a human?" |
| Persona | Agent / employee / individual | Fraud desk, NOC, CISO |
| Build step | None — plain HTML | Vite + React + TypeScript |

## Run

```bash
npm install
npm run dev          # http://localhost:5173, proxying /v1 to the API on :8000
```

The backend must be running (`make run` from the repo root). Then start a stream from
`/console` or any client, and it appears here within a second.

## Build

```bash
npm run build        # type-checks, then emits dist/
npm run preview
```

Set `VITE_API_BASE` to point at a backend on another origin (see `.env.example`).
