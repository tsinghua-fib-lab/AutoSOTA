use log::{LevelFilter, Metadata, Record};
use std::sync::Once;

static INIT_LOGGER: Once = Once::new();
struct SimpleLogger;
static LOGGER: SimpleLogger = SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let module = record.target();
            println!("[{}] [{}] {}", module, record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

pub fn init_logging(default_level: &str) {
    // Only install the logger once
    INIT_LOGGER.call_once(|| {
        log::set_logger(&LOGGER).expect("failed to install SimpleLogger");
    });

    // Always update the log level (can be called multiple times)
    let lvl = default_level
        .parse::<LevelFilter>()
        .unwrap_or(LevelFilter::Info);
    log::set_max_level(lvl);
}
