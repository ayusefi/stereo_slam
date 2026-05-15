import json
import collections

accepted = []
reject_reasons = collections.Counter()
specific_range = []
graph_vertices = []
highest_lc_count = 0

with open('/tmp/sslam_full_ratio10_loops.jsonl', 'r') as f:
    for line in f:
        try:
            record = json.loads(line)
            if record.get('status') == 'accepted':
                accepted.append(record)
            else:
                reason = record.get('reject_reason', 'unknown')
                reject_reasons[reason] += 1
            
            q_id = record.get('query_kf_id')
            c_id = record.get('candidate_kf_id')
            if (q_id is not None and 846 <= q_id <= 858) or (c_id is not None and 5 <= c_id <= 15):
                specific_range.append(record)
            
            if record.get('graph_vertices', 0) > 0:
                graph_vertices.append(record)
                
            processed_lc = record.get('processed_lc_count', 0)
            if processed_lc > highest_lc_count:
                highest_lc_count = processed_lc
        except json.JSONDecodeError:
            continue

print(f"Accepted Count: {len(accepted)}")
print("Accepted Records:")
for r in accepted: print(json.dumps(r))

print("\nReject Reason Counts:")
for reason, count in reject_reasons.items():
    print(f"{reason}: {count}")

print("\nSpecific Range Records (846-858 or 5-15):")
for r in specific_range: print(json.dumps(r))

print("\nGraph Vertices > 0 Records:")
for r in graph_vertices: print(json.dumps(r))

print(f"\nHighest Processed LC Count: {highest_lc_count}")
