#include "algorithms/ments/dbments_logger.h"

using namespace std;

/**
 * Logger entry default implementation
*/
namespace mcts {
    DBMentsLoggerEntry::DBMentsLoggerEntry(
        chrono::duration<double> runtime, int num_visits, int num_backups, double soft_value, double dp_value) : 
            MentsLoggerEntry(runtime, num_visits, num_backups, soft_value), dp_value(dp_value) {}

    void DBMentsLoggerEntry::write_header_to_ostream(ostream& os) {
        os << "runtime," 
            << "num_visits," 
            << "num_backups,"
            << "soft_value,"
            << "dp_value";
    }

    void DBMentsLoggerEntry::write_to_ostream(ostream& os) {
        os << runtime.count() << ","
            << num_visits << ","
            << num_backups << "," 
            << soft_value << ","
            << dp_value;
    }
}

/**
 * Logger default implementation
*/
namespace mcts {

    DBMentsLogger::DBMentsLogger() : 
        MentsLogger() {}
    
    void DBMentsLogger::add_origin_entry() {
        entries.push_back(DBMentsLoggerEntry(
            chrono::duration<double>::zero(), 
            0,
            0,
            0.0,
            0.0));
    }
    
    void DBMentsLogger::log(shared_ptr<MctsDNode> node) {
        DBMentsDNode& dbments_node = (DBMentsDNode&) *node;
        entries.push_back(DBMentsLoggerEntry(
            get_current_total_runtime(), 
            dbments_node.num_visits,
            dbments_node.MentsDNode::num_backups,
            dbments_node.soft_value,
            dbments_node.dp_value));
    }
}