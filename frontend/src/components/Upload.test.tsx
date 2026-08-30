// @vitest-environment jsdom
// Upload screen tests: a picked file becomes a POST, a success links into the new draw's screens, and a validation failure surfaces the shared error panel.

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, uploadDraw } from '../api/client';
import type { UploadResponse } from '../api/types';
import Upload from './Upload';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, uploadDraw: vi.fn() };
});

const uploadDrawMock = vi.mocked(uploadDraw);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

const OK_RESULT: UploadResponse = {
  tournament_id: 'upload-abc123',
  name: 'My Upload Open',
  surface: 'Clay',
  best_of: 3,
  final_set_tiebreak: '7pt_at_6_6',
  draw_size: 8,
  is_simulatable: true,
  placeholder_count: 0,
  is_forecast: false,
  ephemeral: true,
  content_note: 'User-submitted draw. It has not been checked against any record.',
};

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]');
  if (input === null) throw new Error('no file input');
  return input as HTMLInputElement;
}

function pickFile(container: HTMLElement, content: string): void {
  const file = new File([content], 'draw.json', { type: 'application/json' });
  fireEvent.change(fileInput(container), { target: { files: [file] } });
}

it('POSTs the file text and links into the new draw on success', async () => {
  uploadDrawMock.mockResolvedValue(OK_RESULT);
  const { container } = render(<Upload />);

  pickFile(container, '{"draw":"json"}');

  await waitFor(() => screen.getByText('My Upload Open'));
  expect(uploadDrawMock).toHaveBeenCalledWith('{"draw":"json"}');

  const play = screen.getByRole('link', { name: 'Play it out' });
  expect(play.getAttribute('href')).toBe('?tournament=upload-abc123&view=storybook');
  const bracket = screen.getByRole('link', { name: 'View bracket' });
  expect(bracket.getAttribute('href')).toBe('?tournament=upload-abc123&view=bracket');

  // The server's own content note is shown, not a paraphrase, it already carries the ephemeral-storage caveat, so the screen adds no second copy.
  expect(screen.getByText(OK_RESULT.content_note)).toBeTruthy();
});

it('shows a forecast badge for a future-dated upload', async () => {
  uploadDrawMock.mockResolvedValue({ ...OK_RESULT, is_forecast: true });
  const { container } = render(<Upload />);
  pickFile(container, '{}');
  await waitFor(() => screen.getByText('forecast'));
});

it('offers only the bracket, and explains why, for a non-simulatable draw', async () => {
  uploadDrawMock.mockResolvedValue({
    ...OK_RESULT,
    is_simulatable: false,
    placeholder_count: 2,
  });
  const { container } = render(<Upload />);
  pickFile(container, '{}');

  await waitFor(() => screen.getByText('View bracket'));
  expect(screen.queryByRole('link', { name: 'Play it out' })).toBeNull();
  const note = document.querySelector('[data-note="not-simulatable"]');
  expect(note?.textContent).toMatch(/2 placeholder slots/i);
});

it('surfaces the validator problem list on a malformed upload', async () => {
  const error = new ApiError({
    status: 422,
    statusText: 'Unprocessable Entity',
    url: '/tournaments/upload',
    body: {
      detail: {
        reason: 'draw_invalid',
        message: 'Uploaded draw failed validation with 2 problem(s).',
        source: '<uploaded draw>',
        problems: ["surface must be one of ['Clay', 'Grass', 'Hard']", '2 name(s) did not resolve'],
      },
    },
    message: 'failed',
  });
  uploadDrawMock.mockRejectedValue(error);
  const { container } = render(<Upload />);

  pickFile(container, '{"bad":true}');

  await waitFor(() => screen.getByRole('alert'));
  const panel = screen.getByRole('alert');
  expect(panel.getAttribute('data-error-kind')).toBe('draw-invalid');
  expect(panel.textContent).toMatch(/did not resolve/);
});

it('accepts a dropped file, not only a picked one', async () => {
  uploadDrawMock.mockResolvedValue(OK_RESULT);
  render(<Upload />);

  const file = new File(['{"dropped":1}'], 'draw.json', { type: 'application/json' });
  const zone = document.querySelector('[data-dragging]');
  if (zone === null) throw new Error('no drop zone');
  fireEvent.drop(zone, { dataTransfer: { files: [file] } });

  await waitFor(() => screen.getByText('My Upload Open'));
  expect(uploadDrawMock).toHaveBeenCalledWith('{"dropped":1}');
});

describe('accessibility', () => {
  it('the file input is a real, labelled control in the tab order', () => {
    const { container } = render(<Upload />);
    const input = fileInput(container);
    // A real file input, not hidden from assistive tech, that accepts JSON.
    expect(input.getAttribute('type')).toBe('file');
    expect(input.hasAttribute('hidden')).toBe(false);
    expect(input.getAttribute('accept')).toContain('json');
  });
});
