/**
 * Wire types for the ace API — a direct mirror of `src/api/schemas.py`.
 *
 * Every interface here corresponds one-to-one to a Pydantic model in that file,
 * field for field and name for name. Three rules kept it honest, and keeping
 * them is how it stays honest:
 *
 *   * **Mirror the schema, not the docs.** These were written against
 *     `api/schemas.py` and `api/main.py`, which are the contract; prose
 *     elsewhere paraphrases it and drifts.
 *   * **Every schema model sets `extra="forbid"`,** so a response body carries
 *     exactly the declared keys — no optional-unknown escape hatch is needed,
 *     and an extra field on the wire is a server bug rather than something a
 *     client should tolerate.
 *   * **Open string fields stay open; closed ones close.** `surface`,
 *     `best_of`, `final_set_tiebreak`, `mode` and `strategy` are typed as
 *     unions because a validator on the server enumerates their values. Round
 *     labels are *not*: they are a function of the draw size, so anything keyed
 *     by them is `Record<string, …>`.
 */

// --------------------------------------------------------------------------
// Value domains — closed on the server, so closed here.
// --------------------------------------------------------------------------

/** `config.VALID_SURFACES`. `Carpet`/unknown never reach the wire. */
export type Surface = 'Hard' | 'Clay' | 'Grass';

/** `config.VALID_BEST_OF`. */
export type BestOf = 3 | 5;

/** Keys of `config.FINAL_SET_TB_TARGET` (`config.VALID_FINAL_SET_TIEBREAKS`). */
export type FinalSetTiebreak = '7pt_at_6_6' | '10pt_at_6_6' | 'advantage';

/** `sim.reconcile` reconciliation modes; anything else is rejected server-side. */
export type ReconcileMode = 'blend' | 'classifier_anchor';

/** `common.names.MatchStrategy` values, as echoed by `/players`. */
export type MatchStrategy = 'exact' | 'initials' | 'substring' | 'fuzzy';

/**
 * A round label — `QF`/`SF`/`F` for an 8 draw, `R128 … F` for a 128 draw.
 *
 * An alias for `string` on purpose: labels are derived from the draw size
 * (`sim.tournament.round_labels`), so hardcoding a union would be wrong for
 * every draw size but one. Read them off `SimulationResponse.round_labels`.
 */
export type RoundLabel = string;

// --------------------------------------------------------------------------
// /health, /players (T3.1)
// --------------------------------------------------------------------------

/** `api.schemas.HealthResponse`. */
export interface HealthResponse {
  /** `"ok"` once the app has loaded its state. */
  status: string;
  /** Latest season *present in the loaded data* — not `config.END_YEAR`. */
  data_through_year: number;
  n_players: number;
}

/** `api.schemas.SurfaceSkill`. */
export interface SurfaceSkill {
  /** Serve-points-won rate, in `[0, 1]`. */
  spw: number;
  /** Return-points-won rate, in `[0, 1]`. */
  rpw: number;
  /** Raw serve points behind `spw`. **0 means this is the surface baseline**, not a measurement. */
  n_serve_pts: number;
  /** Raw return points behind `rpw`. Same 0 signal. */
  n_return_pts: number;
}

/** `api.schemas.PlayerSummary`. */
export interface PlayerSummary {
  /** Canonical join key; `null` only when the name carries no id in the skill table. */
  player_id: string | null;
  name: string;
  /** Every surface is present for every player (`api.main._player_summary` fills all three). */
  skills: Record<Surface, SurfaceSkill>;
}

/**
 * `api.schemas.PlayerSearchResponse`.
 *
 * All three resolver outcomes arrive as a 200 with this one shape: a unique
 * match is one entry, an ambiguous one is *every* candidate, and no match is
 * `players: []` / `count: 0` / `strategy: null`. Only a missing, blank or
 * whitespace-only query fails, with a 422.
 */
export interface PlayerSearchResponse {
  /** The query as searched — whitespace-stripped by validation, echoed back. */
  query: string;
  count: number;
  /** Which strategy matched; `null` when nothing matched. */
  strategy: MatchStrategy | null;
  players: PlayerSummary[];
}

// --------------------------------------------------------------------------
// /tournaments, /tournaments/{id}/bracket (T3.2)
// --------------------------------------------------------------------------

/** `api.schemas.TournamentSummary` — note it carries no `final_set_tiebreak`. */
export interface TournamentSummary {
  tournament_id: string;
  name: string;
  surface: Surface;
  best_of: BestOf;
  draw_size: number;
}

/** `api.schemas.InvalidDraw` — a draw file that failed to load, listed not skipped. */
export interface InvalidDraw {
  /** The file's stem: a failed file has no trustworthy id of its own. Still addressable. */
  tournament_id: string;
  /** Draw file basename. */
  source: string;
  /** Every problem the draw validator found, in its own order. */
  problems: string[];
}

/** `api.schemas.TournamentListResponse`. */
export interface TournamentListResponse {
  /** Number of **valid** tournaments — `invalid` is counted separately. */
  count: number;
  tournaments: TournamentSummary[];
  invalid: InvalidDraw[];
}

/** `api.schemas.BracketSlot`. */
export interface BracketSlot {
  /** 1-based. Slots `2k-1` and `2k` meet in round 1. */
  position: number;
  player: string;
  /** True for `Qualifier`/`Bye`/… — such a draw is listed and its bracket served, but not simulatable. */
  is_placeholder: boolean;
  /** `null` for placeholders. */
  player_id: string | null;
  seed: number | null;
}

/** `api.schemas.BracketResponse`. */
export interface BracketResponse {
  tournament_id: string;
  name: string;
  surface: Surface;
  best_of: BestOf;
  final_set_tiebreak: FinalSetTiebreak;
  /** Equals `slots.length`. */
  draw_size: number;
  /** Every slot in `position` order — placeholders flagged, never dropped. */
  slots: BracketSlot[];
}

// --------------------------------------------------------------------------
// The shared disclosure (T3.3/T3.4/T3.5)
// --------------------------------------------------------------------------

/**
 * `api.schemas.ModelDisclosure` — the base of both metadata blocks.
 *
 * Every field is required on the server precisely so probabilities cannot be
 * published without their caveat. T4.3/T4.4 must render `is_forecast` and
 * `classifier_limitation`, not merely receive them.
 */
export interface ModelDisclosure {
  /** How `P_clf` and the point model were fused. */
  mode: ReconcileMode;
  /** Weight on the classifier in `blend` mode; unused by `classifier_anchor`. */
  w: number;
  data_through_year: number;
  /** Class name of the pinned classifier (a best-of-four, so it can change). */
  estimator_class: string;
  /** Dotted name of the `ClassifierProb` adapter that produced `P_clf`. */
  adapter: string;
  /** **Machine-readable honesty flag.** `false` → do not present as a prediction. */
  is_forecast: boolean;
  /** One-line, plain-language summary of the model's known limitation. Render it. */
  classifier_limitation: string;
  /** The full technical account behind `classifier_limitation`. Render it behind a collapsed toggle. */
  classifier_limitation_detail: string;
  /** Draw file basename the simulation ran against. */
  source: string;
}

/** `api.schemas.SimulationMetadata` — disclosure plus what only a cached aggregate has. */
export interface SimulationMetadata extends ModelDisclosure {
  runs: number;
  /** Base **RNG** seed, not a player's tournament seed. */
  seed: number;
  /** Processes used. Provenance only — the aggregate is worker-count-independent. */
  workers: number;
  /** ISO-8601 UTC timestamp of when the cache file was written. */
  generated_at: string;
}

/** `api.schemas.StorybookMetadata` — disclosure plus the seed that replays this story. */
export interface StorybookMetadata extends ModelDisclosure {
  /** The **RNG** seed that ran (the caller's, or the server default). */
  seed: number;
  /**
   * Where the *draw* came from: `curated` for a verified example, `user_upload`
   * for a submitted one. Distinct from `is_forecast`, which is about the model.
   */
  content_source: 'curated' | 'user_upload';
  /** Plain-language provenance note — for an upload, that it is unverified and temporary. */
  content_note: string;
}

// --------------------------------------------------------------------------
// /tournaments/{id}/simulate (T3.3)
// --------------------------------------------------------------------------

/** `api.schemas.SimulationPlayer`. */
export interface SimulationPlayer {
  position: number;
  player: string;
  player_id: string | null;
  /** The player's **tournament seed**, unrelated to `SimulationMetadata.seed`. */
  seed: number | null;
  titles: number;
  matches_won: number;
  /** `titles / runs`. */
  p_title: number;
  /** `matches_won / runs`. */
  expected_rounds_won: number;
  /**
   * Round label → runs in which the entrant contested that round.
   *
   * Keyed by **the draw's own labels** (`SimulationResponse.round_labels`), so
   * an 8 draw and a 128 draw return different key sets.
   */
  reached: Record<RoundLabel, number>;
  /** Round label → P(contests that round). Same keying as `reached`. */
  p_reach: Record<RoundLabel, number>;
}

/** `api.schemas.SimulationResponse` — also the on-disk cache format. */
export interface SimulationResponse {
  tournament_id: string;
  name: string;
  surface: Surface;
  best_of: BestOf;
  final_set_tiebreak: FinalSetTiebreak;
  /** The **full** field size — exceeds `count` when `?top=` truncated the rows. */
  draw_size: number;
  /** The draw's round labels, first round first. Derive table columns from this. */
  round_labels: RoundLabel[];
  /** Length of `players`. */
  count: number;
  /** Sorted by title probability, descending (ties broken by bracket position). */
  players: SimulationPlayer[];
  metadata: SimulationMetadata;
}

// --------------------------------------------------------------------------
// /tournaments/{id}/storybook (T3.4)
// --------------------------------------------------------------------------

/** `api.schemas.StorybookMatch`. */
export interface StorybookMatch {
  /** 0-based round, first round first. */
  round_index: number;
  round_label: RoundLabel;
  /** 0-based position within the round, top of the draw first. */
  match_index: number;
  winner: string;
  loser: string;
  winner_seed: number | null;
  loser_seed: number | null;
  /** Winner-first, e.g. `6-4 3-6 7-6(5) 6-2`. */
  scoreline: string;
  /** `"<winner> def. <loser> <scoreline>"`. */
  line: string;
}

/** `api.schemas.StorybookRound`. */
export interface StorybookRound {
  index: number;
  label: RoundLabel;
  matches: StorybookMatch[];
}

/** `api.schemas.StorybookPlayerRun`. */
export interface StorybookPlayerRun {
  position: number;
  player: string;
  player_id: string | null;
  /** Tournament seed. */
  seed: number | null;
  is_champion: boolean;
  /** Last round *contested* — a beaten finalist and the champion both read `F`. */
  furthest_round: RoundLabel;
  matches_won: number;
  beat: string[];
  /** `null` **iff** `is_champion`. */
  eliminated_by: string | null;
}

/** `api.schemas.StorybookChampion`. */
export interface StorybookChampion {
  position: number;
  player: string;
  player_id: string | null;
  seed: number | null;
}

/** `api.schemas.StorybookResponse` — one bracket played out, for one seed. */
export interface StorybookResponse {
  tournament_id: string;
  name: string;
  surface: Surface;
  best_of: BestOf;
  final_set_tiebreak: FinalSetTiebreak;
  draw_size: number;
  /** `draw_size - 1` for a full bracket. */
  match_count: number;
  rounds: StorybookRound[];
  champion: StorybookChampion;
  /** Every entrant's run, best finish first. */
  runs: StorybookPlayerRun[];
  metadata: StorybookMetadata;
}

// --------------------------------------------------------------------------
// POST /tournaments/upload (draw upload feature)
// --------------------------------------------------------------------------

/**
 * `api.schemas.UploadResponse` — acknowledgement that a draw was accepted.
 *
 * `tournament_id` is a generated `upload-…` id in a namespace separate from the
 * curated draws; use it for `/bracket` and `/storybook`. Uploaded draws are
 * **ephemeral** (`ephemeral: true`) — held in memory only, cleared on a server
 * restart — so a shared link may stop working, which the UI discloses.
 */
export interface UploadResponse {
  /** The `upload-…` id to address this draw by. */
  tournament_id: string;
  name: string;
  surface: Surface;
  best_of: BestOf;
  final_set_tiebreak: FinalSetTiebreak;
  draw_size: number;
  /** False when the draw still holds placeholder slots — it lists but cannot run. */
  is_simulatable: boolean;
  placeholder_count: number;
  /** True when the draw's `event_date` is in the future — a genuine forecast. */
  is_forecast: boolean;
  /** Always true: uploaded draws do not survive a restart. */
  ephemeral: boolean;
  /** Note that this draw is user-submitted, unverified and temporary. */
  content_note: string;
}

// --------------------------------------------------------------------------
// Error bodies
// --------------------------------------------------------------------------
// These are not Pydantic response models — they are what FastAPI serialises
// `HTTPException.detail` into. They are typed anyway because a later UI ticket
// has to render them, and the alternative (parsing prose out of a message) is
// what `detail.reason` exists to avoid. Note the asymmetry, faithfully mirrored
// below: the 404 detail is a bare string, the invalid-draw-file 422 carries no
// `reason`, and every other structured body does.

/** One entry of FastAPI's own request-validation 422 body (blank `?query=`, `?top=0`). */
export interface RequestValidationProblem {
  type: string;
  loc: (string | number)[];
  msg: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
  url?: string;
}

/** 422 — a draw file that failed `sim.draw` validation. Carries no `reason`. */
export interface InvalidDrawFileDetail {
  message: string;
  tournament_id: string;
  source: string;
  /** The validator's full problem list, so a hand-entered draw is fixable in one pass. */
  problems: string[];
}

/** 409 — `reason: "draw_not_simulatable"`: the draw still holds placeholder slots. */
export interface NotSimulatableDetail {
  reason: 'draw_not_simulatable';
  message: string;
  tournament_id: string;
  source: string;
  placeholder_slots: { position: number; player: string }[];
}

/** 425 — `reason: "cache_missing"`: nothing precomputed yet. `command` is runnable as-is. */
export interface CacheMissingDetail {
  reason: 'cache_missing';
  message: string;
  tournament_id: string;
  /** The exact precompute command that produces the missing file. Show it. */
  command: string;
}

/** 422 — `reason: "cache_unreadable" | "cache_stale"`: the cache file needs regenerating. */
export interface CacheProblemDetail {
  reason: 'cache_unreadable' | 'cache_stale';
  message: string;
  tournament_id: string;
}

/** 404 — `reason: "upload_not_found"`: an upload id that is no longer held (evicted/restarted). */
export interface UploadNotFoundDetail {
  reason: 'upload_not_found';
  message: string;
  tournament_id: string;
}

/** 422 — `reason: "draw_invalid"`: an uploaded draw failed validation. Carries the problem list. */
export interface UploadDrawInvalidDetail {
  reason: 'draw_invalid';
  message: string;
  source: string;
  /** The validator's full problem list, so a hand-entered draw is fixable in one pass. */
  problems: string[];
}

/**
 * The other structured upload failures, all sharing `{reason, message}`:
 * `unsupported_media_type` (415), `payload_too_large` (413), `invalid_json`
 * (422), `invalid_event_date` (422), `monte_carlo_unavailable_for_upload` (409)
 * and `rate_limited` (429).
 */
export interface UploadProblemDetail {
  reason:
    | 'unsupported_media_type'
    | 'payload_too_large'
    | 'invalid_json'
    | 'invalid_event_date'
    | 'monte_carlo_unavailable_for_upload'
    | 'rate_limited';
  message: string;
  tournament_id?: string;
}

/**
 * Every `detail` **this API** can send: the bare-string 404, the
 * request-validation list, and the four structured bodies.
 *
 * Closed on purpose, so a `switch` over it is exhaustive and a new server-side
 * error shape is a compile error here rather than a silent `default` branch.
 * It deliberately does **not** include `unknown` — a union containing `unknown`
 * collapses to `unknown`, which would make every member above decorative and
 * every exhaustiveness check vacuous.
 *
 * The price is that it does not describe what a *proxy* or a crashed server
 * sends, which is why {@link ApiError.detail} is typed `unknown` rather than
 * this: what arrived on the wire is genuinely not known until it is narrowed.
 * Use `hasReason()` for the `reason`-bearing bodies, and this type to annotate
 * anything that has already been narrowed.
 */
export type ApiErrorDetail =
  | string
  | RequestValidationProblem[]
  | InvalidDrawFileDetail
  | NotSimulatableDetail
  | CacheMissingDetail
  | CacheProblemDetail
  | UploadNotFoundDetail
  | UploadDrawInvalidDetail
  | UploadProblemDetail;

/** The `reason` codes the API attaches to a structured error body. */
export type ApiErrorReason =
  | 'draw_not_simulatable'
  | 'cache_missing'
  | 'cache_unreadable'
  | 'cache_stale'
  | 'upload_not_found'
  | 'draw_invalid'
  | 'unsupported_media_type'
  | 'payload_too_large'
  | 'invalid_json'
  | 'invalid_event_date'
  | 'monte_carlo_unavailable_for_upload'
  | 'rate_limited';
