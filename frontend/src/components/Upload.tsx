/**
 * Upload a tournament draw, then jump straight into playing it out.
 *
 * The one write path in the app. A visitor drops (or picks) a `.json` file, the
 * client POSTs its text verbatim to `/tournaments/upload`, and the server runs
 * the same validation a curated draw goes through. On success the draw gets a
 * generated `upload-…` id, and this screen links to that id's bracket and
 * storybook — normal deep links, so the rest of the app needs no upload-aware
 * code.
 *
 * **Honest about ephemerality, up front and again after.** Uploaded draws live
 * in the server's memory only (there is no persistent disk on the free tier), so
 * they do not survive a restart and a shared link to one can stop working. That
 * is stated as a standing note above the drop zone and repeated on the result,
 * rather than implied away — the same reason the API marks every upload
 * `ephemeral: true`.
 *
 * **Errors reuse the shared {@link ErrorPanel}.** A malformed draw comes back as
 * the validator's accumulated problem list, surfaced the same way every other
 * draw-loading failure in the app is.
 */

import { useCallback, useRef, useState } from 'react';

import { uploadDraw } from '../api/client';
import type { UploadResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import styles from './upload.module.css';

/**
 * A complete, valid eight-slot draw, copied verbatim from the "Minimal worked
 * example" in `data/draws/DRAW_SCHEMA.md`. It is the schema reference's own
 * example, reused rather than re-authored, so this snippet cannot drift from the
 * document a test keeps honest against the validator.
 */
const SCHEMA_EXAMPLE = `{
  "note": "Toy 8-slot illustration of the draw schema — not a real event.",
  "tournament_id": "toy_open_2026",
  "name": "Toy Open 2026: Men's Singles (8-slot example)",
  "surface": "Hard",
  "best_of": 5,
  "final_set_tiebreak": "10pt_at_6_6",
  "event_date": "2026-06-01",
  "draw_size": 8,
  "seeds": {
    "Carlos Alcaraz": 1,
    "Jannik Sinner": 2,
    "Alexander Zverev": 3,
    "Novak Djokovic": 4
  },
  "bracket": [
    { "position": 1, "player": "Carlos Alcaraz" },
    { "position": 2, "player": "Taylor Fritz" },
    { "position": 3, "player": "Alexander Zverev" },
    { "position": 4, "player": "Casper Ruud" },
    { "position": 5, "player": "Novak Djokovic" },
    { "position": 6, "player": "Qualifier" },
    { "position": 7, "player": "Daniil Medvedev" },
    { "position": 8, "player": "Jannik Sinner" }
  ]
}`;

type State =
  | { status: 'idle' }
  | { status: 'uploading' }
  | { status: 'done'; result: UploadResponse }
  | { status: 'error'; error: unknown };

/** A deep link to one view of a tournament id. */
function viewHref(tournamentId: string, view: 'bracket' | 'storybook'): string {
  return `?tournament=${encodeURIComponent(tournamentId)}&view=${view}`;
}

export default function Upload() {
  const [state, setState] = useState<State>({ status: 'idle' });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = useCallback(async (file: File) => {
    setState({ status: 'uploading' });
    let text: string;
    try {
      text = await file.text();
    } catch (error) {
      setState({ status: 'error', error });
      return;
    }
    try {
      const result = await uploadDraw(text);
      setState({ status: 'done', result });
    } catch (error) {
      setState({ status: 'error', error });
    }
  }, []);

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Let the same file be re-picked after a reset by clearing the input value.
    event.target.value = '';
    if (file) void submit(file);
  };

  const onDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void submit(file);
  };

  const reset = () => {
    setState({ status: 'idle' });
    setDragging(false);
  };

  const result = state.status === 'done' ? state.result : null;

  return (
    <section className={styles.upload} aria-label="Upload a draw">
      <p className={styles.intro}>
        Upload your own tournament draw as a JSON file, in the same format as the built-in draws,
        then play it out point by point. Seeds and entrants are yours; the model does the rest.
      </p>

      <p className={styles.ephemeralNote} data-note="ephemeral">
        Uploaded draws are held in memory only. They are cleared when the server restarts, and a
        link you share may stop working later, so save any results you want to keep. Uploads are
        not checked against any official record.
      </p>

      <details className={styles.schema}>
        <summary>See a valid draw file</summary>
        <div className={styles.schemaBody}>
          <p className={styles.schemaLede}>
            Here is a complete eight-slot draw in the format the uploader expects. The full
            field-by-field reference lives in <code>data/draws/DRAW_SCHEMA.md</code>.
          </p>
          <pre className={styles.schemaCode}>
            <code>{SCHEMA_EXAMPLE}</code>
          </pre>
        </div>
      </details>

      {state.status !== 'done' && (
        <label
          className={`${styles.dropzone} ${dragging ? styles.dropzoneActive : ''}`}
          data-dragging={dragging}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <span className={styles.dropTitle}>
            {state.status === 'uploading' ? 'Uploading…' : 'Drop a draw file here'}
          </span>
          <span className={styles.dropHint}>or choose a .json file</span>
          <input
            ref={inputRef}
            id="draw-file"
            className={styles.fileInput}
            type="file"
            accept="application/json,.json"
            disabled={state.status === 'uploading'}
            onChange={onFileChange}
          />
        </label>
      )}

      {state.status === 'error' && <ErrorPanel error={state.error} tournamentId="your upload" />}

      {result !== null && (
        <div className={styles.result} role="status">
          <h2 className={styles.resultHeading}>
            {result.name}
            {result.is_forecast && (
              <span className={styles.badge} data-badge="forecast">
                forecast
              </span>
            )}
          </h2>
          <p className={styles.resultMeta}>
            {result.draw_size}-slot {result.surface} draw · best of {result.best_of} · id{' '}
            {result.tournament_id}
          </p>
          <p className={styles.resultNote}>{result.content_note}</p>

          <div className={styles.actions}>
            {result.is_simulatable ? (
              <>
                <a className={styles.actionPrimary} href={viewHref(result.tournament_id, 'storybook')}>
                  Play it out
                </a>
                <a className={styles.actionSecondary} href={viewHref(result.tournament_id, 'bracket')}>
                  View bracket
                </a>
              </>
            ) : (
              <a className={styles.actionSecondary} href={viewHref(result.tournament_id, 'bracket')}>
                View bracket
              </a>
            )}
          </div>

          {!result.is_simulatable && (
            <p className={styles.resultNote} data-note="not-simulatable">
              This draw still has {result.placeholder_count} placeholder{' '}
              {result.placeholder_count === 1 ? 'slot' : 'slots'} (like Qualifier or Bye), so it
              cannot be simulated. Fill every slot with a real entrant and upload again.
            </p>
          )}

          <button type="button" className={styles.reset} onClick={reset}>
            Upload another draw
          </button>
        </div>
      )}
    </section>
  );
}
