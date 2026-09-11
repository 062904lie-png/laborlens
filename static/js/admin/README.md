# Admin code map

The authenticated panel has seven main sections. Existing controllers remain separate and are exposed as tabs by navigation.js; routes and DOM IDs are retained for compatibility.

| Page | Controller |
| --- | --- |
| Dashboard | dashboard.js |
| Knowledge Base — Documents, Categories, Coverage, FAQs | documents-view.js, knowledge-base.js, upload-wizard.js, categories.js, maintenance.js |
| Answer Reviews — Review Queue, Corrections, Test Chatbot | reviews.js, maintenance.js |
| Reports — Overview, Feedback, Performance | analytics.js, user-feedback.js, maintenance.js |
| Users | users.js, user-access.js |
| Admin Activity | activity-logs.js |
| Settings | settings.js |

Shared modules: core.js (auth/API/document helpers), ui.js (icons/charts/tables/drawers), pages-shared.js (registry), navigation.js (seven-section shell/global search/hash aliases), exports.js (retained download helpers), startup.js (session restoration). admin-simplified.css layers the new layout over the existing theme.

Edit deploy/static/css/admin-reference.css for authenticated layout. Login styling stays in admin-login.css; extracted base styles remain admin-base.css. deploy/admin.html loads modules in dependency order.

Retired controllers faqs.js, unresolved.js, system-health.js, conversations.js, legal-source-review.js and ../admin-workbench.js are not loaded. They are retained as legacy source, not hidden pages.

Legacy conversations/reported-answers hashes open Answer Reviews; source-review/legal-source-review open the Knowledge Base attention filter; performance/system-health open Reports → Performance. The reports hash now opens Reports → Overview. Other prior page IDs open the corresponding tab. Conversation endpoints and stored sessions are untouched.

Backend additions are in backend/routers/admin.py: bounded review/context and feedback/audit pagination, review counts, real stored-event analytics. Configuration remains allowlisted/read-only; browser table density is editable. Administrator API authorization, masked identities, escaped text, safe source links, and audited review/access writes remain enforced.

Run tests/ui/admin_final_test.cjs, tests/ui/admin_panel_test.cjs and backend/tests/test_admin_workbench.py. tests/ui/admin_fixture_server.js is an isolated localhost-only fixture, never a production data source. Full implementation notes: docs/admin-final-implementation.md.
