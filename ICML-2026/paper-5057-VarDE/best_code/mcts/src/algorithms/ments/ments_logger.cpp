#include "algorithms/ments/ments_logger.h"

using namespace std;

/**
 * Logger entry default implementation
*/
namespace mcts {
    MentsLoggerEntry::MentsLoggerEntry(
        chrono::duration<double> runtime, int num_visits, int num_backups, double soft_value) : 
            LoggerEntry(runtime, num_visits), num_backups(num_backups), soft_value(soft_value) {}

    void MentsLoggerEntry::write_header_to_ostream(ostream& os) {
        os << "runtime," 
            << "num_visits," 
            << "num_backups,"
            << "soft_value";
    }

    void MentsLoggerEntry::write_to_ostream(ostream& os) {
        os << runtime.count() << ","
            << num_visits << ","
            << num_backups << "," 
            << soft_value;
    }
}

/**
 * Logger default implementation
*/
namespace mcts {

    MentsLogger::MentsLogger() : 
        MctsLogger() {}
    
    void MentsLogger::add_origin_entry() {
        entries.push_back(MentsLoggerEntry(
            chrono::duration<double>::zero(), 
            0,
            0,
            0.0));
    }
    
    void MentsLogger::log(shared_ptr<MctsDNode> node) {
        MentsDNode& ments_node = (MentsDNode&) *node;
        entries.push_back(MentsLoggerEntry(
            get_current_total_runtime(), 
            ments_node.num_visits,
            ments_node.num_backups,
            ments_node.soft_value));
    }
}