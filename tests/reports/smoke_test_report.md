# Smoke Test Report

**Date**: 2026-06-25 12:40:04  
**Overall**: PASSED  
**Total**: 29 | **Passed**: 29 | **Failed**: 0 | **Errors**: 0 | **Skipped**: 0

## Passed

- `[PASS]` tests/test_mcp_server.py::test_predict_summary_fields
- `[PASS]` tests/test_mcp_server.py::test_predict_total_orders_positive
- `[PASS]` tests/test_mcp_server.py::test_predict_has_delayed_orders
- `[PASS]` tests/test_mcp_server.py::test_predict_delayed_order_has_required_keys
- `[PASS]` tests/test_mcp_server.py::test_predict_formatted_stats_present
- `[PASS]` tests/test_mcp_server.py::test_diagnosis_no_upstream_error
- `[PASS]` tests/test_mcp_server.py::test_diagnosis_has_overall_kpis
- `[PASS]` tests/test_mcp_server.py::test_diagnosis_has_dimension_data
- `[PASS]` tests/test_mcp_server.py::test_simulate_returns_text
- `[PASS]` tests/test_mcp_server.py::test_simulate_reports_rows_affected
- `[PASS]` tests/test_mcp_server.py::test_simulate_mentions_filter_region
- `[PASS]` tests/test_pydantic_models.py::test_row_enrichment_valid
- `[PASS]` tests/test_pydantic_models.py::test_row_enrichment_rejects_short_insights
- `[PASS]` tests/test_pydantic_models.py::test_top_entry_valid
- `[PASS]` tests/test_pydantic_models.py::test_delivery_delay_summary_valid
- `[PASS]` tests/test_pydantic_models.py::test_delivery_delay_summary_with_top_entries
- `[PASS]` tests/test_pydantic_models.py::test_prediction_result_valid
- `[PASS]` tests/test_pydantic_models.py::test_diagnosis_high_risk_valid
- `[PASS]` tests/test_pydantic_models.py::test_diagnosis_comparison_valid
- `[PASS]` tests/test_pydantic_models.py::test_delay_diagnosis_result_valid
- `[PASS]` tests/test_pydantic_models.py::test_simulate_delays_valid
- `[PASS]` tests/test_pydantic_models.py::test_simulations_list_valid
- `[PASS]` tests/test_rag_knowledge.py::test_sla_knowledge_file_exists
- `[PASS]` tests/test_rag_knowledge.py::test_vectorstore_directory_exists
- `[PASS]` tests/test_rag_knowledge.py::test_collection_is_non_empty
- `[PASS]` tests/test_rag_knowledge.py::test_collection_has_reasonable_chunk_count
- `[PASS]` tests/test_rag_knowledge.py::test_peek_returns_documents
- `[PASS]` tests/test_rag_knowledge.py::test_sla_chunks_contain_sla_keywords
- `[PASS]` tests/test_rag_knowledge.py::test_where_document_filter
