use rusqlite::{Connection, Result as SqliteResult};
use serde_json;

use crate::logging::types::{
    CombinedGridSnapshot, ComponentStateSnapshot, EpochScalingSnapshot, FComponentStats,
    GridErrorVariant, LoggingConfig, LoggingMessage, SplitEvent,
};
use crate::logging::EvoLogger;

#[derive(Debug)]
pub struct SqliteBufferedLogger {
    pub(crate) config: LoggingConfig,
    run_id: Option<i64>,
    params_json: String,
    n_rows: usize,
    n_cols: usize,
    split_events: Vec<(usize, usize, SplitEvent)>, // (epoch, tree_id, event)
    combined_grids: Vec<CombinedGridSnapshot>,
    epoch_scalings: Vec<EpochScalingSnapshot>,
    component_states: Vec<(usize, usize, ComponentStateSnapshot)>, // (epoch, tree_id, snapshot)
    f_component_stats: Vec<(usize, usize, usize, FComponentStats, FComponentStats)>, // (epoch, tree_id, iter_no, stats_plus, stats_minus)
    grid_errors: Vec<LoggingMessage>, // GridErrCombined | GridErrFitted variants
    combination_choices: Vec<(usize, String, Option<usize>, Vec<(usize, f64)>)>, // (epoch, method, best_index, candidates)
    conn: Option<Connection>,
    schema_initialized: bool,
}

impl SqliteBufferedLogger {
    pub fn new(config: LoggingConfig) -> Self {
        Self {
            config,
            run_id: None,
            params_json: String::new(),
            n_rows: 0,
            n_cols: 0,
            split_events: Vec::new(),
            combined_grids: Vec::new(),
            epoch_scalings: Vec::new(),
            component_states: Vec::new(),
            f_component_stats: Vec::new(),
            grid_errors: Vec::new(),
            combination_choices: Vec::new(),
            conn: None,
            schema_initialized: false,
        }
    }

    fn apply_pragmas(conn: &Connection) -> SqliteResult<()> {
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;
        conn.pragma_update(None, "cache_size", 100_000)?;
        let _ = conn.execute("PRAGMA mmap_size=1073741824", []);
        Ok(())
    }

    fn init_schema(conn: &Connection) -> SqliteResult<()> {
        conn.pragma_update(None, "journal_mode", "WAL")?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                label TEXT,
                params_json TEXT NOT NULL,
                n_rows INTEGER NOT NULL,
                n_cols INTEGER NOT NULL,
                epochs INTEGER,
                n_trees_per_epoch INTEGER,
                run_type TEXT DEFAULT 'single_grid_tensor' CHECK(run_type IN ('single_grid_tensor', 'tsl_boosted')),
                notes TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER,
                tree_id INTEGER,
                iter_no INTEGER NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('split','resplit','merge')),
                col INTEGER NOT NULL,
                left_interval_idx INTEGER NOT NULL,
                split_value REAL,
                update_a REAL,
                update_b REAL,
                left_count INTEGER,
                right_count INTEGER,
                gain REAL,
                err_before REAL,
                err_after REAL,
                n_cells_after INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS residual_updates (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER,
                tree_id INTEGER,
                iter_no INTEGER NOT NULL,
                sample_id INTEGER NOT NULL,
                residual REAL NOT NULL,
                y_hat REAL,
                PRIMARY KEY (run_id, epoch, tree_id, iter_no, sample_id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS residual_update_blobs (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER,
                tree_id INTEGER,
                iter_no INTEGER NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch, tree_id, iter_no)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS error_reduction_summaries (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER,
                tree_id INTEGER,
                iter_no INTEGER NOT NULL,
                col INTEGER NOT NULL,
                min_val REAL NOT NULL,
                max_val REAL NOT NULL,
                mean_val REAL NOT NULL,
                nan_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                PRIMARY KEY (run_id, epoch, tree_id, iter_no, col)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS combined_grids (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                energy REAL,
                scaling REAL,
                scaling_plus REAL,
                scaling_minus REAL,
                grid_json TEXT NOT NULL,
                f_plus_json TEXT,
                f_minus_json TEXT,
                PRIMARY KEY (run_id, epoch)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS epoch_scalings (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                scaling REAL NOT NULL,
                optimization_epoch INTEGER NOT NULL,
                PRIMARY KEY (run_id, epoch, optimization_epoch)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS component_states (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                tree_id INTEGER NOT NULL,
                iter_no INTEGER NOT NULL,
                col INTEGER NOT NULL,
                data BLOB NOT NULL,
                intervals_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch, tree_id, iter_no, col)
            )",
            [],
        )?;

        // Migrate component_states table to add two-tensor fields (backward compatible)
        Self::migrate_component_states_schema(conn)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS f_component_stats (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                tree_id INTEGER NOT NULL,
                iter_no INTEGER NOT NULL,
                component TEXT NOT NULL CHECK(component IN ('f_plus', 'f_minus')),
                min_val REAL NOT NULL,
                max_val REAL NOT NULL,
                mean_val REAL NOT NULL,
                std_val REAL,
                p25 REAL,
                p50 REAL,
                p75 REAL,
                p95 REAL,
                p99 REAL,
                n_samples INTEGER NOT NULL,
                PRIMARY KEY (run_id, epoch, tree_id, iter_no, component)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS training_errors (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER,
                tree_id INTEGER,
                err REAL NOT NULL,
                created_at TEXT NOT NULL,
                variant TEXT NOT NULL DEFAULT 'train' CHECK(variant IN ('train','test')),
                PRIMARY KEY (run_id, epoch, tree_id, variant)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS combination_choices (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                method TEXT NOT NULL,
                best_index INTEGER,
                candidates_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch, method)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS combination_candidates (
                run_id INTEGER NOT NULL REFERENCES runs(id),
                epoch INTEGER NOT NULL,
                tree_id INTEGER NOT NULL,
                candidate_score REAL NOT NULL,
                is_best INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch, tree_id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_iter ON events(run_id, iter_no)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_action_col ON events(run_id, action, col)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_residual_updates_run_iter ON residual_updates(run_id, iter_no)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_residual_updates_run_sample ON residual_updates(run_id, sample_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_error_summaries_run_iter ON error_reduction_summaries(run_id, iter_no)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_component_states_loc ON component_states(run_id, epoch, tree_id, col, iter_no)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_f_stats_run_iter ON f_component_stats(run_id, epoch, tree_id, iter_no)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_f_stats_component ON f_component_stats(component)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_combination_candidates_run_epoch ON combination_candidates(run_id, epoch)",
            [],
        )?;

        Ok(())
    }

    /// Migrate component_states table to add two-tensor fields (backward compatible)
    /// This function is idempotent - safe to call multiple times
    fn migrate_component_states_schema(conn: &Connection) -> SqliteResult<()> {
        // SQLite doesn't support "IF NOT EXISTS" for ALTER TABLE ADD COLUMN
        // So we catch errors and ignore them (column already exists)
        let _ = conn.execute(
            "ALTER TABLE component_states ADD COLUMN backbone_data BLOB",
            [],
        );
        let _ = conn.execute("ALTER TABLE component_states ADD COLUMN tilt_data BLOB", []);
        let _ = conn.execute(
            "ALTER TABLE component_states ADD COLUMN lambda_plus REAL",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE component_states ADD COLUMN lambda_minus REAL",
            [],
        );
        Ok(())
    }
}

impl EvoLogger for SqliteBufferedLogger {
    fn start_run(&mut self, params_json: &str, n_rows: usize, n_cols: usize) -> i64 {
        self.params_json = params_json.to_string();
        self.n_rows = n_rows;
        self.n_cols = n_cols;

        if self.conn.is_none() {
            let conn = Connection::open(&self.config.db_path)
                .expect("Failed to open sqlite database for logging");
            Self::apply_pragmas(&conn).expect("Failed to apply pragmas");
            Self::init_schema(&conn).expect("Failed to init schema");
            self.schema_initialized = true;
            self.conn = Some(conn);
        }

        let conn = self.conn.as_ref().expect("connection must be initialized");
        let now = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap();

        let run_id = conn
            .prepare_cached(
                "INSERT INTO runs (created_at, label, params_json, n_rows, n_cols, epochs, n_trees_per_epoch, run_type)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING id",
            )
            .and_then(|mut stmt| {
                stmt.query_row(
                    (
                        &now,
                        &self.config.run_label,
                        &self.params_json,
                        self.n_rows as i64,
                        self.n_cols as i64,
                        None::<i64>,
                        None::<i64>,
                        "single_grid_tensor",
                    ),
                    |row| row.get::<_, i64>(0),
                )
            })
            .expect("Failed to insert run");

        self.run_id = Some(run_id);
        run_id
    }

    fn push_split_event(&mut self, epoch: usize, tree_id: usize, event: SplitEvent) {
        self.split_events.push((epoch, tree_id, event));
    }

    fn push_combined_grid(&mut self, snapshot: CombinedGridSnapshot) {
        self.combined_grids.push(snapshot);
    }

    fn push_epoch_scaling(&mut self, snapshot: EpochScalingSnapshot) {
        self.epoch_scalings.push(snapshot);
    }

    fn push_component_state(
        &mut self,
        epoch: usize,
        tree_id: usize,
        snapshot: ComponentStateSnapshot,
    ) {
        self.component_states.push((epoch, tree_id, snapshot));
    }

    fn push_grid_error_combined(&mut self, epoch: usize, err: f64, variant: GridErrorVariant) {
        self.grid_errors.push(LoggingMessage::GridErrCombined {
            epoch,
            err,
            variant,
        });
    }

    fn push_grid_error_fitted(
        &mut self,
        epoch: usize,
        tree_id: usize,
        err: f64,
        variant: GridErrorVariant,
    ) {
        self.grid_errors.push(LoggingMessage::GridErrFitted {
            epoch,
            tree_id,
            err,
            variant,
        });
    }

    fn push_f_component_stats(
        &mut self,
        epoch: usize,
        tree_id: usize,
        iter_no: usize,
        stats_plus: FComponentStats,
        stats_minus: FComponentStats,
    ) {
        self.f_component_stats
            .push((epoch, tree_id, iter_no, stats_plus, stats_minus));
    }

    fn push_combination_choice(
        &mut self,
        epoch: usize,
        method: &str,
        best_index: Option<usize>,
        candidate_indices: Vec<(usize, f64)>,
    ) {
        self.combination_choices
            .push((epoch, method.to_string(), best_index, candidate_indices));
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // No global buffers to drain anymore; events are pushed directly into this instance

        if self.split_events.is_empty()
            && self.combined_grids.is_empty()
            && self.epoch_scalings.is_empty()
            && self.component_states.is_empty()
            && self.f_component_stats.is_empty()
            && self.grid_errors.is_empty()
        {
            return Ok(());
        }

        let conn = self
            .conn
            .as_ref()
            .ok_or("logger connection not initialized")?;
        let tx = conn.unchecked_transaction()?;
        let run_id = self
            .run_id
            .ok_or("run_id not set; call start_run before flush")?;

        // events
        let mut event_stmt = tx.prepare_cached(
            "INSERT INTO events (
                run_id, epoch, tree_id, iter_no, action, col, left_interval_idx, split_value,
                update_a, update_b, left_count, right_count, gain, err_before, err_after,
                n_cells_after, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        )?;
        let now = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)?;
        for (epoch, tree_id, event) in &self.split_events {
            event_stmt.execute(rusqlite::params![
                run_id,
                *epoch as i64,
                *tree_id as i64,
                event.iter_no as i64,
                event.action.as_str(),
                event.col as i64,
                event.left_interval_idx as i64,
                event.split_value,
                event.update_a,
                event.update_b,
                event.left_count.map(|x| x as i64),
                event.right_count.map(|x| x as i64),
                event.gain,
                event.err_before,
                event.err_after,
                event.n_cells_after as i64,
                &now,
            ])?;
        }
        drop(event_stmt);

        // residual updates
        if self.config.record_residual_updates && self.config.pack_updates_as_blob {
            let mut blob_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO residual_update_blobs (run_id, epoch, tree_id, iter_no, data)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (epoch, tree_id, event) in &self.split_events {
                if event.residual_updates.is_empty() {
                    continue;
                }
                let updates = event.residual_updates.clone();
                let data = serde_json::to_string(&updates)?;
                blob_stmt.execute((
                    run_id,
                    *epoch as i64,
                    *tree_id as i64,
                    event.iter_no as i64,
                    data,
                ))?;
            }
            drop(blob_stmt);
        } else if self.config.record_residual_updates {
            let mut residual_stmt = tx.prepare_cached(
                "INSERT INTO residual_updates (run_id, epoch, tree_id, iter_no, sample_id, residual, y_hat)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for (epoch, tree_id, event) in &self.split_events {
                for update in &event.residual_updates {
                    let y_hat_opt: Option<f64> = update.y_hat.map(|x| x as f64);
                    residual_stmt.execute((
                        run_id,
                        *epoch as i64,
                        *tree_id as i64,
                        event.iter_no as i64,
                        update.sample_id as i64,
                        update.residual as f64,
                        y_hat_opt,
                    ))?;
                }
            }
            drop(residual_stmt);
        }

        // combined grids
        if !self.combined_grids.is_empty() {
            let mut cg_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO combined_grids (run_id, epoch, energy, scaling, scaling_plus, scaling_minus, grid_json, f_plus_json, f_minus_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for cg in &self.combined_grids {
                // Serialize f+ and f- arrays to JSON if available
                let f_plus_json = cg
                    .f_plus
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_default());
                let f_minus_json = cg
                    .f_minus
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_default());

                cg_stmt.execute((
                    run_id,
                    cg.epoch as i64,
                    cg.energy,
                    cg.scaling,
                    cg.scaling_plus,
                    cg.scaling_minus,
                    &cg.grid_json,
                    f_plus_json.as_deref(),
                    f_minus_json.as_deref(),
                ))?;
            }
            drop(cg_stmt);
        }

        // epoch scalings
        if !self.epoch_scalings.is_empty() {
            let mut es_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO epoch_scalings (run_id, epoch, scaling, optimization_epoch)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for es in &self.epoch_scalings {
                es_stmt.execute((
                    run_id,
                    es.epoch as i64,
                    es.scaling,
                    es.optimization_epoch as i64,
                ))?;
            }
            drop(es_stmt);
        }

        // component states
        if !self.component_states.is_empty() {
            let mut cs_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO component_states (run_id, epoch, tree_id, iter_no, col, data, intervals_count, created_at, backbone_data, tilt_data, lambda_plus, lambda_minus)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?;
            let now = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)?;
            for (epoch, tree_id, cs) in &self.component_states {
                cs_stmt.execute(rusqlite::params![
                    run_id,
                    *epoch as i64,
                    *tree_id as i64,
                    cs.iter_no as i64,
                    cs.col as i64,
                    &cs.data,
                    cs.intervals_count as i64,
                    &now,
                    cs.backbone_data.as_deref(),
                    cs.tilt_data.as_deref(),
                    cs.lambda_plus,
                    cs.lambda_minus,
                ])?;
            }
            drop(cs_stmt);
        }

        // f_component_stats
        if !self.f_component_stats.is_empty() {
            let mut f_stats_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO f_component_stats (run_id, epoch, tree_id, iter_no, component, min_val, max_val, mean_val, std_val, p25, p50, p75, p95, p99, n_samples)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            )?;
            for (epoch, tree_id, iter_no, stats_plus, stats_minus) in &self.f_component_stats {
                // Insert f_plus stats
                f_stats_stmt.execute(rusqlite::params![
                    run_id,
                    *epoch as i64,
                    *tree_id as i64,
                    *iter_no as i64,
                    "f_plus",
                    stats_plus.min,
                    stats_plus.max,
                    stats_plus.mean,
                    stats_plus.std,
                    stats_plus.p25,
                    stats_plus.p50,
                    stats_plus.p75,
                    stats_plus.p95,
                    stats_plus.p99,
                    stats_plus.n_samples as i64,
                ])?;
                // Insert f_minus stats
                f_stats_stmt.execute(rusqlite::params![
                    run_id,
                    *epoch as i64,
                    *tree_id as i64,
                    *iter_no as i64,
                    "f_minus",
                    stats_minus.min,
                    stats_minus.max,
                    stats_minus.mean,
                    stats_minus.std,
                    stats_minus.p25,
                    stats_minus.p50,
                    stats_minus.p75,
                    stats_minus.p95,
                    stats_minus.p99,
                    stats_minus.n_samples as i64,
                ])?;
            }
            drop(f_stats_stmt);
        }

        // combination choices (JSON summary + normalized candidates)
        if !self.combination_choices.is_empty() {
            let mut cc_summary_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO combination_choices (run_id, epoch, method, best_index, candidates_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut cc_candidate_delete = tx.prepare_cached(
                "DELETE FROM combination_candidates WHERE run_id = ?1 AND epoch = ?2",
            )?;
            let mut cc_candidate_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO combination_candidates (run_id, epoch, tree_id, candidate_score, is_best, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let now = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)?;
            for (epoch, method, best_index, candidates) in &self.combination_choices {
                // Serialize candidates as JSON array of {tree_id, score}
                let v: Vec<_> = candidates
                    .iter()
                    .map(|(tid, score)| serde_json::json!({ "tree_id": *tid, "score": *score }))
                    .collect();
                let candidates_json = serde_json::to_string(&v)?;

                // Insert summary row
                cc_summary_stmt.execute(rusqlite::params![
                    run_id,
                    *epoch as i64,
                    method,
                    best_index.map(|b| b as i64),
                    candidates_json,
                    &now
                ])?;

                // Remove any previous candidate rows for this epoch/run, then insert fresh ones
                cc_candidate_delete.execute((run_id, *epoch as i64))?;
                for (tree_id, score) in candidates {
                    let is_best = if Some(*tree_id) == *best_index {
                        1i64
                    } else {
                        0i64
                    };
                    cc_candidate_stmt.execute(rusqlite::params![
                        run_id,
                        *epoch as i64,
                        *tree_id as i64,
                        *score,
                        is_best,
                        &now
                    ])?;
                }
            }
            drop(cc_candidate_stmt);
            drop(cc_candidate_delete);
            drop(cc_summary_stmt);
        }

        // grid errors (train/test)
        if !self.grid_errors.is_empty() {
            let mut te_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO training_errors (run_id, epoch, tree_id, err, created_at, variant)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let now = time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)?;
            for msg in &self.grid_errors {
                match msg {
                    LoggingMessage::GridErrCombined {
                        epoch,
                        err,
                        variant,
                    } => {
                        te_stmt.execute(rusqlite::params![
                            run_id,
                            *epoch as i64,
                            None::<i64>, // tree_id is None for combined
                            err,
                            &now,
                            variant.as_str(),
                        ])?;
                    }
                    LoggingMessage::GridErrFitted {
                        epoch,
                        tree_id,
                        err,
                        variant,
                    } => {
                        te_stmt.execute(rusqlite::params![
                            run_id,
                            *epoch as i64,
                            Some(*tree_id as i64),
                            err,
                            &now,
                            variant.as_str(),
                        ])?;
                    }
                    _ => {} // Ignore non-grid-error messages
                }
            }
            drop(te_stmt);
        }

        tx.commit()?;
        self.split_events.clear();
        self.combined_grids.clear();
        self.component_states.clear();
        self.f_component_stats.clear();
        self.grid_errors.clear();
        self.combination_choices.clear();
        Ok(())
    }
}
