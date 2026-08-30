// The one write path: POST a .json draw verbatim to /tournaments/upload (server-side validation). On success the draw gets an `upload-...` id linked as normal deep links; errors reuse the shared ErrorPanel.

import { useCallback, useRef, useState } from 'react';

import { uploadDraw } from '../api/client';
import type { UploadResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import styles from './upload.module.css';

/** The real, placeholder-free draw a reader can open to see the exact format. */
const EXAMPLE_DRAW_URL =
  'https://github.com/hahalaa/ace/blob/main/data/draws/ausopen_2026_atp_full.json';

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
    event.target.value = ''; // let the same file be re-picked after a reset

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
        Upload your own tournament draw as a JSON file following the same format as the built-in
        draws, then play it out point by point.
      </p>

      <p className={styles.schemaLink}>
        For the exact format, open a{' '}
        <a href={EXAMPLE_DRAW_URL} target="_blank" rel="noreferrer">
          real draw file on GitHub
        </a>{' '}
        or read the <code>data/draws/DRAW_SCHEMA.md</code> field reference.
      </p>

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
