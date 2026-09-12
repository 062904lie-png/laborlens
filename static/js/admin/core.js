// ════════════════════════════════════════════════════════════════
// CONFIGURATION — Update NGROK_URL here when your ngrok URL changes
// ════════════════════════════════════════════════════════════════
const NGROK_URL = 'https://plaster-multiply-tint.ngrok-free.dev/api';
// ════════════════════════════════════════════════════════════════

let API = NGROK_URL;   // Set immediately — no login input needed
let token = null, currentUser = null, allUsers = [], faqCache = [];
let docContentState = { id: null, offset: 0, limit: 25, total: 0 };
let docObjectUrl = null;
let kbDocumentsCache = [];
let kbSearchableChunks = null;
let kbSelectedCategory = 'All';
let kbDocumentSearch = '';
let kbAdvancedFilters = {primary:'All',secondary:'All',issuer:'All',type:'All',status:'All'};
let kbSelectedDocumentIds = new Set();
let editingFaqId = null;
const ADMIN_SESSION_KEY = 'laborlens_admin_session';
function adminPageSize(){const value=Number(localStorage.getItem('laborlens_admin_page_size')||10);return [7,10,20,50].includes(value)?value:10;}
function adminReportPeriod(){return localStorage.getItem('laborlens_default_report_period')==='7'?'7':'30';}
const DOC_VERSION_STATUSES = ['current', 'active', 'superseded', 'historical', 'under_review', 'amended', 'repealed', 'archived', 'inactive', 'replaced', 'outdated'];
const KB_CATEGORIES = [
  'Core Labor Laws',
  'Employment Benefits',
  'Leave Benefits',
  'Domestic Workers',
  'Anti-Harassment / Discrimination',
  'Social Benefits',
  'Occupational Safety & Health',
  'DOLE Resources',
  'Labor Relations',
  'Jurisprudence',
  'Labor Advisories',
  'Others',
  'Civil Service / Government Employment'
];

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, ch => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[ch]));
}

function jsArg(v) {
  return esc(String(v ?? '').replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/\r?\n/g,' '));
}

function num(v, fallback=0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function pct(part, whole) {
  const total = num(whole);
  return total > 0 ? Math.round((num(part) / total) * 100) : 0;
}

function dateLabel(v) {
  if (!v) return '-';
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? String(v).slice(0, 10) : d.toLocaleDateString();
}

function languageLabel(code) {
  return ({ en: 'English', fil: 'Filipino', hil: 'Hiligaynon' }[String(code || '').toLowerCase()] || 'Unknown');
}

function safeUrl(v) {
  const raw=String(v ?? '').trim();
  if(!raw) return '';
  try {
    const url=new URL(raw);
    return ['http:','https:'].includes(url.protocol) ? url.href : '';
  } catch(_) {
    return '';
  }
}

function docTypeLabel(doc) {
  const type = String(doc?.file_type || '').toLowerCase();
  const name = String(doc?.original_filename || '').toLowerCase();
  if (type === 'pdf' || name.endsWith('.pdf')) return 'PDF';
  if (type === 'docx' || name.endsWith('.docx')) return 'DOCX';
  if (['png','jpg','jpeg','tif','tiff','bmp','webp'].includes(type) || /\.(png|jpe?g|tiff?|bmp|webp)$/.test(name)) return 'IMAGE';
  return type ? type.toUpperCase() : '-';
}

function versionStatusBadge(status='active', isActive=true) {
  const value=String(status||'active').toLowerCase();
  const cls=['active','current'].includes(value)&&isActive?'badge-green':value==='repealed'?'badge-red':['amended','replaced','outdated'].includes(value)?'badge-amber':'badge-gray';
  const label=isActive?value.toUpperCase():`INACTIVE / ${value.toUpperCase()}`;
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function unresolvedSuggestionText(intent='general') {
  const key=String(intent||'general').toLowerCase();
  if(key.includes('leave')) return 'Add a FAQ for leave rules or upload a source under Leave Benefits.';
  if(key.includes('termination') || key.includes('dismiss')) return 'Upload a termination-related source under Core Labor Laws or add a reviewed FAQ.';
  if(key.includes('wage') || key.includes('pay') || key.includes('overtime')) return 'Upload a wage-related source under Employment Benefits or DOLE Resources.';
  if(key.includes('occupational') || key.includes('workplace safety') || key.includes('osh') || key.includes('hazard')) return 'Upload an OSH source under Occupational Safety & Health or add a focused FAQ answer.';
  if(key.includes('harass') || key.includes('safe spaces')) return 'Upload a source under Anti-Harassment / Discrimination or add a focused FAQ answer.';
  return 'Review the question, then add a FAQ or upload a source document under the best matching reference documents category.';
}

function revokeDocObjectUrl() {
  if (docObjectUrl) {
    URL.revokeObjectURL(docObjectUrl);
    docObjectUrl = null;
  }
}

function saveAdminSession() {
  if (!token || !currentUser) return;
  localStorage.setItem(ADMIN_SESSION_KEY, JSON.stringify({
    token,
    user: currentUser,
    savedAt: Date.now()
  }));
}

function clearAdminSession() {
  localStorage.removeItem(ADMIN_SESSION_KEY);
}

function showLogin(message='') {
  closeDocModal(); closeAdminNav();
  document.getElementById('dashboard').classList.remove('visible');
  document.getElementById('login-screen').classList.remove('gone');
  document.getElementById('login-err').textContent = message;
  const pass = document.getElementById('l-pass');
  if (pass) pass.focus();
}

function renderKbCategoryOptions(categories=KB_CATEGORIES, selected=KB_CATEGORIES[0]) {
  const select = document.getElementById('kb-category');
  if (!select) return;
  const safeCategories = Array.isArray(categories) && categories.length ? categories : KB_CATEGORIES;
  select.innerHTML = safeCategories
    .map(c => `<option value="${esc(c)}" ${c===selected?'selected':''}>${esc(c)}</option>`)
    .join('');
  renderSecondaryCategoryOptions('kb-secondary-categories', select.value, getSelectedSecondaryCategories('kb-secondary-categories'));
}

function parseMetadataList(value) {
  if (Array.isArray(value)) return [...new Set(value.map(v=>String(v||'').trim()).filter(Boolean))];
  const raw=String(value||'').trim(); if(!raw) return [];
  if(raw.startsWith('[')){try{const parsed=JSON.parse(raw);if(Array.isArray(parsed))return parseMetadataList(parsed);}catch(_e){}}
  return [...new Set(raw.split(/[,;|]/).map(v=>v.trim()).filter(Boolean))];
}

function getSelectedSecondaryCategories(containerId) {
  return [...document.querySelectorAll(`#${containerId} input[type="checkbox"]:checked`)].map(input=>input.value);
}

function renderSecondaryCategoryOptions(containerId, primaryCategory, selected=[]) {
  const wrap=document.getElementById(containerId); if(!wrap) return;
  const chosen=new Set(parseMetadataList(selected));
  wrap.innerHTML=KB_CATEGORIES.map(category=>{
    const disabled=category===primaryCategory;
    return `<label class="kb-secondary-option${disabled?' disabled':''}"><input type="checkbox" value="${esc(category)}" ${chosen.has(category)&&!disabled?'checked':''} ${disabled?'disabled':''}/><span>${esc(category)}</span></label>`;
  }).join('');
  wrap.onchange=event=>{if(event.target.checked&&getSelectedSecondaryCategories(containerId).length>3){event.target.checked=false;toast('Choose at most three extra categories.',true);}};
}

function syncSecondaryCategoryOptions(containerId, primaryCategory) {
  renderSecondaryCategoryOptions(containerId, primaryCategory, getSelectedSecondaryCategories(containerId));
}

async function suggestUploadCategories() {
  try{
    const payload={
      title:document.getElementById('kb-title')?.value||'',
      law_number:document.getElementById('kb-law-number')?.value||'',
      document_type:document.getElementById('kb-document-type')?.value||'',
      issuing_agency:document.getElementById('kb-issuing-agency')?.value||''
    };
    const r=await apiFetch('/kb/categories/suggest',{method:'POST',body:JSON.stringify(payload)});
    const suggestion=await r.json(); if(!r.ok)throw new Error(suggestion.detail||'Unable to suggest categories.');
    const primary=document.getElementById('kb-category'); if(primary)primary.value=suggestion.primary_category;
    renderSecondaryCategoryOptions('kb-secondary-categories',suggestion.primary_category,suggestion.secondary_categories||[]);
    const note=document.getElementById('kb-category-suggestion');
    if(note)note.textContent=`Suggested for review (${Math.round(Number(suggestion.confidence||0)*100)}%): ${(suggestion.reasons||[]).join('; ')}`;
  }catch(e){toast(e.message,true);}
}

function countByStatus(rows=[], status='indexed') {
  const match = rows.find(r => String(r.status || '').toLowerCase() === status);
  return match ? Number(match.count || 0) : 0;
}

function kbSummary(stats={}) {
  const chunks = Number(stats.chroma?.total_chunks || 0);
  const docs = stats.documents || [];
  const categories = stats.categories || [];
  const indexedDocs = countByStatus(docs, 'indexed');
  const activeCategories = categories.filter(c => Number(c.count || 0) > 0).length;
  return { chunks, indexedDocs, activeCategories };
}

function documentDisplayTitle(doc={}) {
  return doc.document_title || doc.original_filename || doc.filename || 'Untitled document';
}

// Preserve complete names on hover while keeping dense admin views compact.
function documentTitleHtml(value) {
  const title = String(value || 'Untitled document');
  return `<span class="document-title" title="${esc(title)}">${esc(title)}</span>`;
}

function shortDocumentLabel(value, maximum=72) {
  const title = String(value || 'Untitled document');
  return title.length > maximum ? `${title.slice(0, Math.max(1, maximum - 1))}…` : title;
}

function normalizedKbCategory(category='') {
  const raw = String(category || '').trim();
  return KB_CATEGORIES.includes(raw) ? raw : 'Others';
}

function isIndexedKbDocument(doc={}) {
  return String(doc.status || '').toLowerCase() === 'indexed' && Number(doc.chunk_count || 0) > 0;
}

function kbCategoryCounts(docs=kbDocumentsCache) {
  const counts = Object.fromEntries(['All', ...KB_CATEGORIES].map(category => [category, 0]));
  (Array.isArray(docs) ? docs : []).forEach(doc => {
    counts.All += 1;
    const categories=new Set([normalizedKbCategory(doc.primary_category||doc.category),...parseMetadataList(doc.secondary_categories)]);
    categories.forEach(category=>{if(KB_CATEGORIES.includes(category))counts[category]=(counts[category]||0)+1;});
  });
  return counts;
}

function renderKbCategoryFilters() {
  const wrap = document.getElementById('kb-category-tabs');
  if (!wrap) return;
  const counts = kbCategoryCounts();
  const categories = ['All', ...KB_CATEGORIES];
  wrap.innerHTML = categories.map(category => {
    const active = kbSelectedCategory === category ? ' active' : '';
    return `<button type="button" class="kb-filter-chip${active}" onclick="setKbCategoryFilter('${jsArg(category)}')">${esc(category)} <span class="kb-filter-count">${counts[category] || 0}</span></button>`;
  }).join('');
}

function documentMatchesSearch(doc={}, query='') {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return true;
  return [
    documentDisplayTitle(doc),
    doc.original_filename,
    doc.filename,
    doc.category,
    doc.primary_category,
    parseMetadataList(doc.secondary_categories).join(' '),
    doc.document_type,
    doc.legal_role,
    parseMetadataList(doc.topics).join(' '),
    parseMetadataList(doc.sub_intents).join(' '),
    doc.source_name,
    doc.source_url,
    doc.law_number,
    doc.article_section,
    doc.issuing_agency,
    doc.date_issued,
    doc.effective_date,
    doc.date_downloaded,
    doc.superseding_document,
    doc.law_version,
    doc.status,
    doc.processing_stage,
    doc.file_type
  ].some(value => String(value || '').toLowerCase().includes(q));
}

function filteredKbDocuments() {
  return kbDocumentsCache.filter(doc => {
    const primary=normalizedKbCategory(doc.primary_category||doc.category);
    const secondary=parseMetadataList(doc.secondary_categories);
    const allCategories=new Set([primary,...secondary]);
    const inCategory = kbSelectedCategory === 'All' || allCategories.has(kbSelectedCategory);
    const matchesPrimary=kbAdvancedFilters.primary==='All'||primary===kbAdvancedFilters.primary;
    const matchesSecondary=kbAdvancedFilters.secondary==='All'||secondary.includes(kbAdvancedFilters.secondary);
    const matchesIssuer=kbAdvancedFilters.issuer==='All'||String(doc.issuing_agency||'')===kbAdvancedFilters.issuer;
    const matchesType=kbAdvancedFilters.type==='All'||String(doc.document_type||'')===kbAdvancedFilters.type;
    const matchesStatus=kbAdvancedFilters.status==='All'||documentProcessingStatus(doc)===kbAdvancedFilters.status;
    return inCategory&&matchesPrimary&&matchesSecondary&&matchesIssuer&&matchesType&&matchesStatus&&documentMatchesSearch(doc,kbDocumentSearch);
  });
}

function renderKbAdvancedFilters(){
  const optionHtml=(values,selected,label)=>[`<option value="All">${label}</option>`,...values.map(v=>`<option value="${esc(v)}" ${v===selected?'selected':''}>${esc(v)}</option>`)].join('');
  const unique=field=>[...new Set(kbDocumentsCache.map(doc=>String(doc[field]||'').trim()).filter(Boolean))].sort();
  const primary=document.getElementById('kb-primary-filter');if(primary)primary.innerHTML=optionHtml(KB_CATEGORIES,kbAdvancedFilters.primary,'All main categories');
  const secondary=document.getElementById('kb-secondary-filter');if(secondary)secondary.innerHTML=optionHtml(KB_CATEGORIES,kbAdvancedFilters.secondary,'All extra categories');
  const issuer=document.getElementById('kb-issuer-filter');if(issuer)issuer.innerHTML=optionHtml(unique('issuing_agency'),kbAdvancedFilters.issuer,'All issuers');
  const type=document.getElementById('kb-type-filter');if(type)type.innerHTML=optionHtml(unique('document_type'),kbAdvancedFilters.type,'All document types');
  const status=document.getElementById('kb-status-filter');if(status)status.innerHTML=optionHtml(['Pending','Reading text','Splitting text','Preparing for search','Processing','Replacing','Ready','Not ready','Failed'],kbAdvancedFilters.status,'All statuses');
}

function setKbAdvancedFilter(kind,value='All'){
  if(Object.prototype.hasOwnProperty.call(kbAdvancedFilters,kind))kbAdvancedFilters[kind]=value||'All';
  renderDocumentsTable();
}

function updateKbBulkSelection(){
  const count=kbSelectedDocumentIds.size;
  const label=document.getElementById('kb-bulk-selection');if(label)label.textContent=`${count} document${count===1?'':'s'} selected`;
  const button=document.getElementById('kb-bulk-edit-btn');if(button)button.disabled=count===0;
}

function toggleKbDocumentSelection(id,checked){
  if(checked)kbSelectedDocumentIds.add(Number(id));else kbSelectedDocumentIds.delete(Number(id));
  updateKbBulkSelection();
}

function toggleAllVisibleDocuments(checked){
  const rows=window.AdminWorkbench?window.AdminWorkbench.documentPage(filteredKbDocuments()):filteredKbDocuments();
  rows.forEach(doc=>{if(checked)kbSelectedDocumentIds.add(Number(doc.id));else kbSelectedDocumentIds.delete(Number(doc.id));});
  renderDocumentsTable();
}

function openBulkDocumentEdit(){
  if(!kbSelectedDocumentIds.size)return;
  document.getElementById('doc-modal').classList.add('show');
  document.getElementById('doc-modal-title').textContent='Bulk Edit Documents';
  document.getElementById('doc-modal-sub').textContent=`Update document details for ${kbSelectedDocumentIds.size} selected documents. Blank fields remain unchanged.`;
  const categoryOptions=['<option value="">Keep existing primary categories</option>',...KB_CATEGORIES.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`)].join('');
  const statusOptions=['<option value="">Keep existing legal statuses</option>',...DOC_VERSION_STATUSES.map(s=>`<option value="${s}">${esc(s)}</option>`)].join('');
  document.getElementById('doc-modal-body').innerHTML=`<div class="form-grid">
    <div class="form-row"><div class="form-group"><label>Main Category</label><select id="bulk-primary-category" onchange="syncSecondaryCategoryOptions('bulk-secondary-categories',this.value)">${categoryOptions}</select></div><div class="form-group"><label>Legal Status</label><select id="bulk-version-status">${statusOptions}</select></div></div>
    <div class="form-group"><label>Extra Categories</label><div class="kb-secondary-grid" id="bulk-secondary-categories"></div><span class="status-note"><input type="checkbox" id="bulk-set-secondary"/> Replace extra categories with this selection</span></div>
    <div class="form-group"><label>Document Type</label><input id="bulk-document-type" placeholder="Leave blank to keep existing"/></div>
  </div>`;
  renderSecondaryCategoryOptions('bulk-secondary-categories','',[]);
  document.getElementById('doc-modal-foot').innerHTML='<button class="btn btn-sm btn-primary" onclick="saveBulkDocumentEdit()">Apply Bulk Edit</button>';
}

async function saveBulkDocumentEdit(){
  try{
    const primary=document.getElementById('bulk-primary-category')?.value||null;
    const setSecondary=!!document.getElementById('bulk-set-secondary')?.checked;
    const versionStatus=document.getElementById('bulk-version-status')?.value||null;
    const documentType=document.getElementById('bulk-document-type')?.value?.trim()||null;
    const payload={document_ids:[...kbSelectedDocumentIds],primary_category:primary,version_status:versionStatus,document_type:documentType};
    if(setSecondary)payload.secondary_categories=getSelectedSecondaryCategories('bulk-secondary-categories');
    const r=await apiFetch('/kb/bulk/metadata',{method:'PUT',body:JSON.stringify(payload)});
    const result=await r.json();if(!r.ok)throw new Error(result.detail||'Bulk edit failed.');
    toast(`Updated ${result.updated_count||0} documents.`);
    kbSelectedDocumentIds.clear();closeDocModal();await loadDocuments();updateKbBulkSelection();
  }catch(e){toast(e.message,true);}
}

function setKbCategoryFilter(category='All') {
  kbSelectedCategory = category === 'All' || KB_CATEGORIES.includes(category) ? category : 'All';
  renderDocumentsTable();
}

function setKbDocumentSearch(value='') {
  kbDocumentSearch = String(value || '');
  renderDocumentsTable();
}

function updateKbFilterSummary(total=0, shown=0) {
  const el = document.getElementById('kb-filter-summary');
  if (!el) return;
  const categoryText = kbSelectedCategory === 'All' ? 'all categories' : kbSelectedCategory;
  const searchText = kbDocumentSearch.trim() ? ` matching "${kbDocumentSearch.trim()}"` : '';
  el.textContent = `Showing ${shown.toLocaleString()} of ${total.toLocaleString()} documents in ${categoryText}${searchText}`;
}

function renderDocumentsTable() {
  if(window.AdminKnowledge) return AdminKnowledge.render();
  const tb = document.getElementById('docs-tbody');
  if (!tb) return;
  renderKbCategoryFilters();
  renderKbAdvancedFilters();
  const filteredDocs = filteredKbDocuments();
  const docs = window.AdminWorkbench ? window.AdminWorkbench.documentPage(filteredDocs) : filteredDocs;
  updateKbBulkSelection();
  updateKbFilterSummary(kbDocumentsCache.length, filteredDocs.length);
  if (!kbDocumentsCache.length) {
    tb.innerHTML = '<tr><td colspan="9" class="empty">No reference documents yet.</td></tr>';
    return;
  }
  if (!docs.length) {
    tb.innerHTML = '<tr><td colspan="9" class="empty">No documents match this category or search.</td></tr>';
    return;
  }
  tb.innerHTML = docs.map((d,i)=>{
    const processingStage=String(d.processing_stage||'');
    const isReplacing=processingStage.startsWith('replacing');
    const isProcessing=d.status==='processing'||d.status==='pending'||isReplacing;
    const canRetry=!isProcessing;
    const vectorMissing=kbSearchableChunks===0 && d.status==='indexed' && (d.chunk_count??0)>0;
    const shownChunks=vectorMissing ? 0 : (d.chunk_count??0);
    const canView=!!d.filename;
    const displayTitle=esc(d.document_title||d.original_filename);
    const name=esc(d.original_filename);
    const nameArg=jsArg(d.original_filename);
    const category=esc(d.primary_category||d.category||KB_CATEGORIES[0]);
    const secondaryCategories=parseMetadataList(d.secondary_categories);
    const secondaryLabel=secondaryCategories.length?`<span class="kb-secondary-list">Secondary: ${secondaryCategories.map(esc).join(', ')}</span>`:'';
    const docMeta=[d.source_name,d.document_type,d.legal_role,d.law_number,d.article_section,d.issuing_agency,d.date_issued,d.effective_date,d.date_downloaded,d.superseding_document,d.law_version,d.processing_stage,parseMetadataList(d.topics).join(', '),parseMetadataList(d.sub_intents).join(', ')].filter(Boolean).map(esc).join(' | ');
    const sourceUrl=safeUrl(d.source_url);
    const originalLink=sourceUrl?`<span class="status-note"><a href="${esc(sourceUrl)}" target="_blank" rel="noopener noreferrer">${esc(d.source_url)}</a></span>`:'';
    return `<tr>
      <td class="td-mono"><input type="checkbox" aria-label="Select ${displayTitle}" ${kbSelectedDocumentIds.has(Number(d.id))?'checked':''} onchange="toggleKbDocumentSelection(${d.id},this.checked)"/><span class="status-note">${i+1}</span></td>
      <td class="td-strong doc-name"><span class="doc-title-short" title="${displayTitle}">${displayTitle}</span><details class="doc-details"><summary>Details</summary><div><strong>${displayTitle}</strong><span class="status-note">File: ${name}</span>${docMeta?`<span class="status-note">${docMeta}</span>`:''}${originalLink}</div></details>${d.error_message?`<span class="status-note" style="color:var(--red);">${esc(d.error_message)}</span>`:''}</td>
      <td><span class="badge badge-blue kb-category-badge">Primary: ${category}</span>${secondaryLabel}</td>
      <td><span class="badge badge-gray">${docTypeLabel(d)}</span></td>
      <td>${d.file_size?(d.file_size/1048576).toFixed(2)+' MB':'-'}</td>
      <td><strong>${shownChunks}</strong></td>
      <td>${statusBadge(d, vectorMissing)}</td>
      <td class="td-mono">${dateLabel(d.last_indexed_at||d.created_at)}</td>
      <td><div class="doc-actions">
        <button class="btn btn-sm btn-primary" ${canView?'':'disabled'} onclick="viewDoc(${d.id},'${nameArg}')">View Original</button>
        <button class="btn btn-sm btn-primary" ${isProcessing?'disabled title="Wait until indexing or replacement finishes"':''} onclick="replaceDoc(${d.id},'${nameArg}')">Replace</button>
        <button class="btn btn-sm btn-gold" onclick="viewDocHealth(${d.id},'${nameArg}')">Health</button>
        ${canRetry?`<button class="btn btn-sm btn-gold" onclick="if(confirm('Re-index this saved file? Searchable chunks will be rebuilt.')) retryDocumentIndex(${d.id},'${nameArg}')">Re-index</button><button class="btn btn-sm btn-gold" onclick="AdminWorkbench.archive(${d.id})">Archive</button>`:''}
        <button class="btn btn-sm btn-gold" onclick="editDocumentMeta(${d.id})">Edit</button>
        <button class="btn btn-sm btn-danger" ${isProcessing?'disabled title="Wait until indexing finishes"':''} onclick="deleteDoc(${d.id},'${nameArg}','${d.status}')">Delete</button>
      </div></td>
    </tr>`;
  }).join('');
}

function renderHealthRow(label, value, color='green', note='') {
  const bg = color === 'amber' ? 'var(--amber-bg)' : color === 'red' ? 'var(--red-bg)' : color === 'blue' ? 'var(--blue-bg)' : 'var(--green-bg)';
  const border = color === 'amber' ? 'rgba(217,119,6,.2)' : color === 'red' ? 'rgba(192,57,43,.2)' : color === 'blue' ? 'rgba(29,78,216,.2)' : 'rgba(13,122,85,.2)';
  return `<div class="flex-between" style="padding:12px 16px;background:${bg};border:1px solid ${border};border-radius:10px;gap:12px;">
    <span style="font-size:14px;font-weight:600;">${label}${note?`<span class="status-note">${note}</span>`:''}</span>
    <span class="badge badge-${color}">${value}</span>
  </div>`;
}

function toast(msg, err=false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderLeftColor = err ? 'var(--red)' : 'var(--gold)';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}

async function apiFetch(path, opts={}) {
  const response = await fetch(`${API}${path}`, { ...opts, headers: {
    'Content-Type':'application/json','Authorization':`Bearer ${token}`,
    'ngrok-skip-browser-warning':'true',...(opts.headers||{})
  }});
  if (response.status === 401) {
    token = null;
    currentUser = null;
    clearAdminSession();
    showLogin('Your admin session expired. Please sign in again.');
  }
  return response;
}

async function loadKbCategories() {
  renderKbCategoryOptions();
  try {
    const r = await apiFetch('/kb/categories');
    const d = await r.json();
    if (r.ok) renderKbCategoryOptions(d.categories, d.default || KB_CATEGORIES[0]);
  } catch (_) {
    renderKbCategoryOptions();
  }
}

async function restoreAdminSession() {
  renderKbCategoryOptions();
  const raw = localStorage.getItem(ADMIN_SESSION_KEY);
  if (!raw) {
    showLogin();
    return;
  }
  try {
    const saved = JSON.parse(raw);
    if (!saved?.token) throw new Error('No saved session');
    token = saved.token;
    currentUser = saved.user || null;

    const r = await fetch(`${API}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'ngrok-skip-browser-warning': 'true'
      }
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || 'Saved session expired');
    if (d.role !== 'admin') throw new Error('Saved session is not an admin account');

    currentUser = { id: d.id, username: d.username, email: d.email, role: d.role };
    saveAdminSession();
    initDashboard();
  } catch (e) {
    token = null;
    currentUser = null;
    clearAdminSession();
    showLogin('Please sign in to continue.');
  }
}

async function doLogin() {
  const email = document.getElementById('l-email').value.trim();
  const pass  = document.getElementById('l-pass').value;
  const errEl = document.getElementById('login-err');
  if (!email || !pass) {
    errEl.textContent = 'Please enter your email and password.';
    return;
  }
  errEl.textContent = 'Authenticating…';
  try {
    const r = await fetch(`${API}/auth/login`, {
      method:'POST',
      headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({ email, password: pass })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Authentication failed.');
    if (d.role !== 'admin') throw new Error('Access denied: admin account required.');
    token = d.access_token;
    document.getElementById('l-pass').value = '';
    currentUser = { id: d.user_id, username: d.username, role: d.role };
    saveAdminSession();
    initDashboard();
  } catch(e) { errEl.textContent = e.message; }
}

function doLogout() {
  token = null; currentUser = null;
  closeDocModal(); closeAdminNav();
  clearAdminSession();
  showLogin();
  document.getElementById('l-pass').value = '';
}

function initDashboard() {
  document.getElementById('login-screen').classList.add('gone');
  document.getElementById('dashboard').classList.add('visible');
  document.getElementById('tb-user').textContent = currentUser.username;
  document.getElementById('tb-avatar').textContent = currentUser.username.charAt(0).toUpperCase();
  loadKbCategories();
  refreshAll();
}

function refreshAll() {
  loadOverview(); loadUsers(); loadDocuments(); loadAnalytics();
}

function switchTab(el, name) {
  if(window.AdminShell) return AdminShell.go(name);
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.content-inner > [id^="tab-"]').forEach(t => t.classList.remove('tab-active'));
  document.getElementById(`tab-${name}`).classList.add('tab-active');
  closeAdminNav();
  document.getElementById('main').scrollTop = 0;
  window.LaborLensAdminUI?.syncNavigation(name);
  window.AdminWorkbench?.onTab(name);
}

function toggleAdminNav() {
  const nav = document.getElementById('left-nav');
  const overlay = document.getElementById('admin-nav-overlay');
  const btn = document.getElementById('admin-menu-btn');
  const open = nav.classList.toggle('open');
  overlay.classList.toggle('open', open);
  if (btn) btn.setAttribute('aria-expanded', String(open));
}

function closeAdminNav() {
  const nav = document.getElementById('left-nav');
  const overlay = document.getElementById('admin-nav-overlay');
  const btn = document.getElementById('admin-menu-btn');
  if (nav) nav.classList.remove('open');
  if (overlay) overlay.classList.remove('open');
  if (btn) btn.setAttribute('aria-expanded', 'false');
}
