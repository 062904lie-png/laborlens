/* -- Reference Documents -- */
function documentProcessingStatus(d, vectorMissing=false) {
  const state=String(d.status||'pending').toLowerCase();
  const stage=String(d.processing_stage||'').toLowerCase();
  if(state==='error'||state==='failed')return 'Failed';
  if(stage.startsWith('replacing'))return 'Replacing';
  if(state==='indexed')return vectorMissing||Number(d.chunk_count||0)===0?'Not ready':'Ready';
  if(/chunk/.test(stage))return 'Splitting text';
  if(/extract|ocr|pars/.test(stage))return 'Reading text';
  if(/embed|index|vector/.test(stage))return 'Preparing for search';
  if(state==='processing')return 'Processing';
  return 'Pending';
}
function statusBadge(d, vectorMissing=false) {
  const label=documentProcessingStatus(d,vectorMissing);
  return `<span class="badge badge-${label==='Ready'?'green':label==='Failed'?'red':'amber'}">${esc(label)}</span>`;
}
function legacyStatusBadge(d, vectorMissing=false) {
  const s=(d.status||'pending').toLowerCase();
  const stage=String(d.processing_stage||'').toLowerCase();
  if(vectorMissing) return '<span class="badge badge-amber">NEEDS RE-INDEX</span><span class="status-note">Upload again to rebuild searchable sections.</span>';
  if(stage.startsWith('replacing')) return '<span class="badge badge-amber">REPLACING</span><span class="status-note">Validating and indexing replacement...</span>';
  if(s==='indexed') return '<span class="badge badge-green">INDEXED</span>';
  if(s==='error' && String(d.error_message||'').toLowerCase().includes('chunks were cleared')) return '<span class="badge badge-amber">NEEDS RE-INDEX</span><span class="status-note">Upload again to rebuild searchable sections.</span>';
  if(s==='error') return '<span class="badge badge-red">ERROR</span>';
  if(s==='processing') return '<span class="badge badge-amber">PROCESSING</span><span class="status-note">Splitting text and indexing...</span>';
  return `<span class="badge badge-gray">${esc(s.toUpperCase())}</span>`;
}

function sleep(ms){return new Promise(resolve=>setTimeout(resolve,ms));}

let documentRefreshTimer=null,documentRefreshPromise=null,documentRefreshQueued=false;
function loadDocuments() {
  clearTimeout(documentRefreshTimer);
  if(documentRefreshPromise){documentRefreshQueued=true;return documentRefreshPromise;}
  documentRefreshPromise=refreshDocuments().finally(()=>{
    documentRefreshPromise=null;
    const queued=documentRefreshQueued;documentRefreshQueued=false;
    const processing=kbDocumentsCache.some(d=>['pending','processing'].includes(d.status)||String(d.processing_stage||'').startsWith('replacing'));
    if(token&&(queued||processing))documentRefreshTimer=setTimeout(loadDocuments,queued?0:3000);
  });
  return documentRefreshPromise;
}

async function refreshDocuments() {
  const tb=document.getElementById('docs-tbody');
  let searchableChunks=null;
  try {
    const sr=await apiFetch('/kb/stats');
    if(sr.ok){
      const stats=await sr.json();
      const summary=kbSummary(stats);
      searchableChunks=summary.chunks;
      const modelNote=stats.chroma?.embedding_model ? `<span class="status-note">Search model: ${esc(stats.chroma.embedding_model)}</span>` : '';
      const resetNote=stats.chroma?.last_reset_reason ? `<span class="status-note" style="color:var(--amber);">Document search was reset after an update. Upload the documents again to make them searchable.</span>` : '';
      document.getElementById('kb-stats').innerHTML=summary.chunks>0
        ? `<strong>${summary.chunks.toLocaleString()} searchable sections</strong> available across ${summary.activeCategories||1} document ${(summary.activeCategories||1)===1?'category':'categories'}.${modelNote}`
        : `<strong>No searchable sections yet.</strong> Upload a legal document and wait until it is ready to search.${modelNote}${resetNote}`;
      document.getElementById('kb-chunks').textContent=summary.chunks.toLocaleString();
    }
    const dr=await apiFetch('/kb/documents');
    const docs=await dr.json();
    if(!dr.ok) throw new Error(docs.detail||'Failed');
    kbDocumentsCache = Array.isArray(docs) ? docs : [];
    const hasSearchableChunks=searchableChunks===null
      ? Number(document.getElementById('kb-chunks').textContent.replace(/,/g,''))>0
      : searchableChunks>0;
      const readyCount=hasSearchableChunks
        ? docs.filter(d=>d.status==='indexed' && (d.chunk_count??0)>0 && d.is_active && ['active','current'].includes(String(d.version_status||'active').toLowerCase())).length
        : 0;
    document.getElementById('kb-docs-count').textContent=readyCount;
    kbSearchableChunks = searchableChunks;
    renderDocumentsTable();
  } catch(e){tb.innerHTML=`<tr><td colspan="6"><div class="err-msg">Error: ${esc(e.message)} <button class="btn btn-sm" onclick="loadDocuments()">Retry</button></div></td></tr>`;}
}

async function uploadFile(input) {
  const file=input.files[0]; if(!file) return;
  const msg=document.getElementById('upload-msg');
  const category=document.getElementById('kb-category')?.value||KB_CATEGORIES[0];
  msg.innerHTML=`<div style="margin-top:14px;padding:14px;background:var(--amber-bg);color:var(--amber);border-radius:10px;font-size:13px;font-weight:600;">Uploading <strong>${esc(file.name)}</strong> to ${esc(category)}...</div>`;
  try {
    const form=new FormData();
    form.append('file',file);
    form.append('category',category);
    form.append('primary_category',category);
    form.append('secondary_categories',JSON.stringify(getSelectedSecondaryCategories('kb-secondary-categories')));
    form.append('document_title',document.getElementById('kb-title')?.value||'');
    form.append('document_type',document.getElementById('kb-document-type')?.value||'');
    form.append('legal_role',document.getElementById('kb-legal-role')?.value||'');
    form.append('implements',JSON.stringify(parseMetadataList(document.getElementById('kb-implements')?.value||'')));
    form.append('topics',JSON.stringify(parseMetadataList(document.getElementById('kb-topics')?.value||'')));
    form.append('sub_intents',JSON.stringify(parseMetadataList(document.getElementById('kb-sub-intents')?.value||'')));
    form.append('source_name',document.getElementById('kb-source')?.value||'');
    form.append('source_url',document.getElementById('kb-source-url')?.value||'');
    form.append('law_number',document.getElementById('kb-law-number')?.value||'');
    form.append('article_section',document.getElementById('kb-article-section')?.value||'');
    form.append('issuing_agency',document.getElementById('kb-issuing-agency')?.value||'');
    form.append('date_issued',document.getElementById('kb-date-issued')?.value||'');
    form.append('effective_date',document.getElementById('kb-effective-date')?.value||'');
    form.append('date_downloaded',document.getElementById('kb-date-downloaded')?.value||'');
    form.append('superseding_document',document.getElementById('kb-superseding-document')?.value||'');
    form.append('law_version',document.getElementById('kb-version')?.value||'');
    form.append('version_status',document.getElementById('kb-version-status')?.value||'active');
    const r=await fetch(`${API}/kb/upload`,{method:'POST',headers:{'Authorization':`Bearer ${token}`,'ngrok-skip-browser-warning':'true'},body:form});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Upload failed');
    msg.setAttribute('role','status');
    msg.setAttribute('aria-live','polite');
    msg.innerHTML=`<p>Document added successfully. ${documentTitleHtml(file.name)}Processing continues in the background. You can upload another file now. Follow its progress in the Status column.</p>`;
    toast('Upload accepted. Processing in background.');
    // Keep the accepted job visible and polling even if the first refresh fails.
    if(!kbDocumentsCache.some(doc=>Number(doc.id)===Number(d.doc_id))){
      kbDocumentsCache.unshift({id:d.doc_id,document_title:form.get('document_title')||file.name,original_filename:file.name,category,status:'processing',processing_stage:'uploaded',chunk_count:0});
      renderDocumentsTable();
    }
    ['kb-title','kb-document-type','kb-legal-role','kb-implements','kb-topics','kb-sub-intents','kb-source','kb-source-url','kb-law-number','kb-article-section','kb-issuing-agency','kb-date-issued','kb-effective-date','kb-date-downloaded','kb-superseding-document','kb-version'].forEach(id=>{const el=document.getElementById(id); if(el) el.value='';});
    renderSecondaryCategoryOptions('kb-secondary-categories',category,[]);
    const statusField=document.getElementById('kb-version-status'); if(statusField) statusField.value='current';
    loadDocuments();
    input.value='';
    return d;
  } catch(e){
    msg.innerHTML=`<div style="margin-top:14px;padding:14px;background:var(--red-bg);color:var(--red);border-radius:10px;font-size:13px;">${esc(e.message)}</div>`;
    toast(e.message,true);
  }
}

async function waitForDocumentIndexed(docId,fileName,msg) {
  let checks=0;
  msg.setAttribute('role','status');
  msg.setAttribute('aria-live','polite');
  while(true){
    let d;
    try {
      const r=await apiFetch(`/kb/documents/${docId}`);
      if([401,403,404].includes(r.status)){
        const error=new Error(r.status===404?'Document no longer exists.':'Sign in again to check document processing.');
        error.terminal=true;
        throw error;
      }
      if(!r.ok) throw new Error('Status temporarily unavailable');
      d=await r.json();
    } catch(error) {
      if(error.terminal) throw error;
      msg.innerHTML='<div class="status-note" role="status">Connection interrupted. Document processing may still be running; reconnecting automatically…</div>';
      await sleep(5000);
      continue;
    }
    if(d.status==='indexed') return d;
    if(d.status==='error') throw new Error(d.error_message||'Document indexing failed.');
    msg.innerHTML=`<div style="margin-top:14px;padding:14px;background:var(--amber-bg);color:var(--amber);border-radius:10px;font-size:13px;font-weight:600;">Preparing for search <strong>${esc(fileName)}</strong> in ${esc(d.category||KB_CATEGORIES[0])}... ${d.chunk_count??0} sections ready so far.</div>`;
    checks++;
    await sleep(checks<20?1500:5000);
  }
}

async function retryDocumentIndex(docId,fileName='Document') {
  try {
    const r=await apiFetch(`/kb/documents/${docId}/retry-index`,{method:'POST'});
    const d=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(d.detail||'Unable to retry document indexing.');
    toast(`Re-indexing ${fileName} in the background.`);
    loadDocuments();
  } catch(e) {
    toast(e.message,true);
  }
}

function replacementStatusHtml(job={}, fileName='') {
  const status=String(job.status||'queued').toLowerCase();
  const progress=Math.max(0,Math.min(100,num(job.progress,0)));
  const isDone=status==='completed';
  const isFailed=status==='failed';
  const color=isFailed?'red':isDone?'green':'gold';
  const title=isFailed?'Replacement failed':isDone?'Replacement completed':'Replacing document';
  const message=job.message||`Preparing replacement${fileName?` for ${fileName}`:''}...`;
  const details=[
    job.old_chunk_count ? `Old chunks: ${num(job.old_chunk_count).toLocaleString()}` : '',
    job.new_chunk_count ? `New chunks: ${num(job.new_chunk_count).toLocaleString()}` : '',
    job.warning_message ? `Warning: ${job.warning_message}` : '',
    job.error_message ? `Error: ${job.error_message}` : ''
  ].filter(Boolean).map(esc).join('<br/>');
  return `<div class="replace-status-card">
    <div class="replace-status-title"><span>${esc(title)}</span><span>${progress}%</span></div>
    <div class="progress-bar"><div class="progress-fill ${color==='green'?'green':color==='red'?'red':'gold'}" style="width:${progress}%"></div></div>
    <div class="replace-status-msg">${esc(message)}${details?`<br/>${details}`:''}</div>
  </div>`;
}

async function replaceDoc(id,name='Document') {
  revokeDocObjectUrl();
  docContentState={id,offset:0,limit:25,total:0,name:name||'Document'};
  document.getElementById('doc-modal').classList.add('show');
  document.getElementById('doc-modal-title').textContent=`Replace ${name||'Document'}`;
  document.getElementById('doc-modal-sub').textContent='Upload a corrected file. The current document stays active until the replacement is fully indexed.';
  document.getElementById('doc-modal-body').innerHTML=`
    <div class="alert-bar blue"><span>ℹ️</span><div><div style="font-weight:700;">Safe replacement mode</div><div style="font-size:12px;margin-top:2px;">LaborLens will validate and index the new file first. If replacement fails, the original document and its chunks remain active.</div></div></div>
    <div class="form-grid">
      <div class="form-group">
        <label for="doc-replace-file">Replacement file</label>
        <input type="file" id="doc-replace-file" accept=".pdf,.docx,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/*"/>
        <span class="status-note">PDF, DOCX, PNG, JPG, TIFF, BMP, or WEBP. Use the most original corrected document.</span>
      </div>
      <div class="form-group">
        <label for="doc-replace-notes">Replacement notes</label>
        <textarea id="doc-replace-notes" placeholder="Optional: why this document is being replaced"></textarea>
      </div>
      <label style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);"><input type="checkbox" id="doc-replace-preserve" checked/> Keep current category, source link, title, and citation metadata</label>
    </div>
    <div id="doc-replace-status"></div>`;
  document.getElementById('doc-modal-foot').innerHTML=`<button class="btn btn-sm btn-primary" onclick="submitReplacement(${id})">Start Replacement</button><button class="btn btn-sm btn-gold" onclick="viewDoc(${id},'${jsArg(name||'Document')}')">Cancel</button>`;
}

async function submitReplacement(id) {
  const input=document.getElementById('doc-replace-file');
  const statusEl=document.getElementById('doc-replace-status');
  const file=input?.files?.[0];
  if(!file){toast('Choose a replacement file first.',true);return;}
  if(statusEl) statusEl.innerHTML=replacementStatusHtml({status:'queued',progress:0,message:`Uploading ${file.name}...`},file.name);
  const form=new FormData();
  form.append('file',file);
  form.append('replacement_notes',document.getElementById('doc-replace-notes')?.value||'');
  form.append('preserve_metadata',document.getElementById('doc-replace-preserve')?.checked ? 'true' : 'false');
  try{
    const r=await fetch(`${API}/kb/documents/${id}/replace`,{
      method:'POST',
      headers:{'Authorization':`Bearer ${token}`,'ngrok-skip-browser-warning':'true'},
      body:form
    });
    const data=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(data.detail||'Replacement failed to start.');
    document.getElementById('doc-modal-foot').innerHTML=`<button class="btn btn-sm btn-primary" onclick="loadDocuments()">Refresh Table</button><button class="btn btn-sm btn-danger" onclick="closeDocModal()">Close</button>`;
    if(statusEl) statusEl.innerHTML=replacementStatusHtml({status:'queued',progress:1,message:'Replacement queued. Waiting for validation and indexing...'},file.name);
    loadDocuments();
    await pollReplacementJob(data.job_id,id,file.name);
  }catch(e){
    if(statusEl) statusEl.innerHTML=replacementStatusHtml({status:'failed',progress:100,message:'Replacement could not start.',error_message:e.message},file.name);
    toast(e.message,true);
    loadDocuments();
  }
}

async function pollReplacementJob(jobId,docId,fileName='') {
  const statusEl=document.getElementById('doc-replace-status');
  for(let i=0;i<180;i++){
    const r=await apiFetch(`/kb/replacement-jobs/${encodeURIComponent(jobId)}`);
    const job=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(job.detail||'Unable to check replacement status.');
    if(statusEl) statusEl.innerHTML=replacementStatusHtml(job,fileName);
    if(i===0 || i%4===0) loadDocuments();
    const status=String(job.status||'').toLowerCase();
    if(status==='completed'){
      toast(`Replacement indexed with ${job.new_chunk_count??0} chunks.`);
      loadDocuments(); loadOverview(); loadAnalytics();
      document.getElementById('doc-modal-foot').innerHTML=`<button class="btn btn-sm btn-primary" onclick="viewDoc(${docId},'${jsArg(job.new_filename||fileName||'Document')}')">View Original</button><button class="btn btn-sm btn-danger" onclick="closeDocModal()">Close</button>`;
      return job;
    }
    if(status==='failed'){
      toast(job.error_message||'Replacement failed. Original document was kept.',true);
      loadDocuments();
      return job;
    }
    await sleep(1500);
  }
  throw new Error('Replacement is still running. Refresh the document list to check status.');
}

async function deleteDoc(id,name,status='') {
  if(status==='processing'||status==='pending'){toast('Wait until indexing finishes before deleting this document.',true);return;}
  if(!confirm(`This permanently deletes the document, its stored file, extracted content, and all related knowledge-base chunks.\n\nDocument: "${name}"\n\nThis action cannot be undone.`)) return;
  try {
    const r=await apiFetch(`/kb/documents/${id}`,{method:'DELETE'});
    const d=await r.json().catch(()=>({}));
    if(!r.ok) throw new Error(d.detail||'Failed.');
    toast(`Deleted ${d.deleted_chunks??0} chunks and the uploaded file.`);
    loadDocuments(); loadOverview(); loadAnalytics();
  } catch(e){toast(e.message,true);}
}

async function viewDocHealth(id,name='Document Health') {
  revokeDocObjectUrl();
  docContentState={id,offset:0,limit:25,total:0,name:name||'Document Health'};
  document.getElementById('doc-modal').classList.add('show');
  document.getElementById('doc-modal-title').textContent=name||'Document Health';
  document.getElementById('doc-modal-sub').textContent='Processing and indexing status';
  document.getElementById('doc-modal-body').innerHTML='<div class="loading">Loading document health...</div>';
  document.getElementById('doc-modal-foot').innerHTML='';
  try{
    const r=await apiFetch(`/kb/documents/${id}/health`);
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Unable to load document health.');
    const doc=data.document||{};
    const health=data.health||{};
    const stages=health.stages||{};
    const stageCard=(label,key,note='')=>`<div class="mini-metric"><div class="mini-metric-lbl">${label}</div><div class="mini-metric-val" style="font-size:18px;">${stages[key]?'YES':'NO'}</div>${note?`<div class="status-note">${note}</div>`:''}</div>`;
    document.getElementById('doc-modal-sub').textContent=`${doc.category||KB_CATEGORIES[0]} | ${documentProcessingStatus(doc)}`;
    document.getElementById('doc-modal-body').innerHTML=`
      <div class="doc-meta-grid">
        ${stageCard('Uploaded file','uploaded', health.file_exists?'File exists on server':'File missing on server')}
        ${stageCard('Text extracted','text_extracted', `${num(health.text_char_count).toLocaleString()} characters`)}
        ${stageCard('OCR used','ocr_used', `Method: ${esc(String(health.extraction_method||'none').toUpperCase())}`)}
        ${stageCard('Chunked','chunked', `${num(health.chunk_count).toLocaleString()} chunks`)}
        ${stageCard('Embedded','embedded','Added to vector database')}
        ${stageCard('Ready','ready', `Stage: ${esc(health.processing_stage||'uploaded')}`)}
      </div>
      <div class="full-doc-card">
        <div class="full-doc-toolbar">${documentTitleHtml(doc.document_title||doc.original_filename||'Document')}<span>${num(health.page_count)} pages</span></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">
          <div class="mini-metric"><div class="mini-metric-lbl">Status</div><div class="mini-metric-val" style="font-size:18px;">${statusBadge(doc)}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Visibility</div><div class="mini-metric-val" style="font-size:18px;">${doc.is_active?'ACTIVE':'INACTIVE'}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Processing stage</div><div class="mini-metric-val" style="font-size:18px;">${esc(String(doc.processing_stage||'uploaded').toUpperCase())}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Last indexed</div><div class="mini-metric-val" style="font-size:18px;">${esc(dateLabel(doc.last_indexed_at||doc.created_at))}</div></div>
        </div>
      </div>`;
    document.getElementById('doc-modal-foot').innerHTML=`<button class="btn btn-sm btn-primary" onclick="editDocumentMeta(${id})">Edit Document Details</button><button class="btn btn-sm btn-primary" onclick="viewDoc(${id},'${jsArg(doc.original_filename||name)}')">Open Original</button>`;
  }catch(e){
    document.getElementById('doc-modal-body').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
  }
}

async function editDocumentMeta(id) {
  revokeDocObjectUrl();
  docContentState={id,offset:0,limit:25,total:0,name:'Document Details'};
  document.getElementById('doc-modal').classList.add('show');
  document.getElementById('doc-modal-title').textContent='Document Details';
  document.getElementById('doc-modal-sub').textContent='Edit source link, category, and visibility. Processing status updates automatically.';
  document.getElementById('doc-modal-body').innerHTML='<div class="loading">Loading document metadata...</div>';
  document.getElementById('doc-modal-foot').innerHTML='';
  try{
    const r=await apiFetch(`/kb/documents/${id}`);
    const doc=await r.json();
    if(!r.ok) throw new Error(doc.detail||'Unable to load document metadata.');
    const categoryOptions=(KB_CATEGORIES||[]).map(cat=>`<option value="${esc(cat)}" ${cat===(doc.category||KB_CATEGORIES[0])?'selected':''}>${esc(cat)}</option>`).join('');
    const versionOptions=DOC_VERSION_STATUSES.map(status=>`<option value="${status}" ${status===String(doc.version_status||'active').toLowerCase()?'selected':''}>${status.charAt(0).toUpperCase()+status.slice(1)}</option>`).join('');
    document.getElementById('doc-modal-body').innerHTML=`
      <div class="form-grid">
        <div class="form-group"><label>Document Title</label><input type="text" id="doc-edit-title" value="${esc(doc.document_title||'')}"/></div>
        <div class="form-row">
          <div class="form-group"><label>Document Type</label><input type="text" id="doc-edit-document-type" value="${esc(doc.document_type||'')}"/></div>
          <div class="form-group"><label>Legal Role</label><input type="text" id="doc-edit-legal-role" value="${esc(doc.legal_role||'')}"/></div>
        </div>
        <div class="form-group"><label>Implements</label><input type="text" id="doc-edit-implements" value="${esc(parseMetadataList(doc.implements).join(', '))}"/></div>
        <div class="form-row">
          <div class="form-group"><label>Topics</label><input type="text" id="doc-edit-topics" value="${esc(parseMetadataList(doc.topics).join(', '))}"/></div>
          <div class="form-group"><label>Sub-intents</label><input type="text" id="doc-edit-sub-intents" value="${esc(parseMetadataList(doc.sub_intents).join(', '))}"/></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Official Source</label><input type="text" id="doc-edit-source" value="${esc(doc.source_name||'')}"/></div>
          <div class="form-group"><label>Law Year / Version</label><input type="text" id="doc-edit-version" value="${esc(doc.law_version||'')}"/></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Law / Order Number</label><input type="text" id="doc-edit-law-number" value="${esc(doc.law_number||'')}"/></div>
          <div class="form-group"><label>Article / Section</label><input type="text" id="doc-edit-article-section" value="${esc(doc.article_section||'')}"/></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Issued By</label><input type="text" id="doc-edit-issuing-agency" value="${esc(doc.issuing_agency||'')}"/></div>
          <div class="form-group"><label>Date Issued</label><input type="text" id="doc-edit-date-issued" value="${esc(doc.date_issued||'')}"/></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Date the Rule Takes Effect</label><input type="text" id="doc-edit-effective-date" value="${esc(doc.effective_date||'')}"/></div>
          <div class="form-group"><label>Date Downloaded</label><input type="text" id="doc-edit-date-downloaded" value="${esc(doc.date_downloaded||'')}"/></div>
        </div>
        <div class="form-group"><label>Document That Updates or Replaces This</label><input type="text" id="doc-edit-superseding-document" value="${esc(doc.superseding_document||'')}"/></div>
        <div class="form-group"><label>Original Source Link</label><input type="url" id="doc-edit-source-url" value="${esc(doc.source_url||'')}"/></div>
        <div class="form-row">
          <div class="form-group"><label>Main Category</label><select id="doc-edit-category" onchange="syncSecondaryCategoryOptions('doc-edit-secondary-categories',this.value)">${categoryOptions}</select></div>
          <div class="form-group"><label>Status</label><div>${statusBadge(doc)}</div><span class="status-note">Updated automatically during processing.</span><input type="hidden" id="doc-edit-version-status" value="${esc(doc.version_status||'active')}"/></div>
        </div>
        <div class="form-group"><label>Extra Categories</label><div class="kb-secondary-grid" id="doc-edit-secondary-categories"></div></div>
        <label style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);"><input type="checkbox" id="doc-edit-active" ${doc.is_active?'checked':''}/> Active for chatbot retrieval</label>
      </div>`;
    renderSecondaryCategoryOptions('doc-edit-secondary-categories',doc.primary_category||doc.category||KB_CATEGORIES[0],doc.secondary_categories||[]);
    document.getElementById('doc-modal-foot').innerHTML=`<button class="btn btn-sm btn-primary" onclick="saveDocumentMeta(${id})">Save Changes</button><button class="btn btn-sm btn-gold" onclick="viewDocHealth(${id},'${jsArg(doc.original_filename||'Document Health')}')">View Health</button>`;
  }catch(e){
    document.getElementById('doc-modal-body').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
  }
}

async function saveDocumentMeta(id) {
  try{
    const payload={
      document_title: document.getElementById('doc-edit-title')?.value?.trim()||'',
      document_type: document.getElementById('doc-edit-document-type')?.value?.trim()||'',
      legal_role: document.getElementById('doc-edit-legal-role')?.value?.trim()||'',
      implements: parseMetadataList(document.getElementById('doc-edit-implements')?.value||''),
      topics: parseMetadataList(document.getElementById('doc-edit-topics')?.value||''),
      sub_intents: parseMetadataList(document.getElementById('doc-edit-sub-intents')?.value||''),
      source_name: document.getElementById('doc-edit-source')?.value?.trim()||'',
      source_url: document.getElementById('doc-edit-source-url')?.value?.trim()||'',
      law_number: document.getElementById('doc-edit-law-number')?.value?.trim()||'',
      article_section: document.getElementById('doc-edit-article-section')?.value?.trim()||'',
      issuing_agency: document.getElementById('doc-edit-issuing-agency')?.value?.trim()||'',
      date_issued: document.getElementById('doc-edit-date-issued')?.value?.trim()||'',
      effective_date: document.getElementById('doc-edit-effective-date')?.value?.trim()||'',
      date_downloaded: document.getElementById('doc-edit-date-downloaded')?.value?.trim()||'',
      superseding_document: document.getElementById('doc-edit-superseding-document')?.value?.trim()||'',
      law_version: document.getElementById('doc-edit-version')?.value?.trim()||'',
      category: document.getElementById('doc-edit-category')?.value||KB_CATEGORIES[0],
      primary_category: document.getElementById('doc-edit-category')?.value||KB_CATEGORIES[0],
      secondary_categories: getSelectedSecondaryCategories('doc-edit-secondary-categories'),
      is_active: !!document.getElementById('doc-edit-active')?.checked,
      version_status: document.getElementById('doc-edit-version-status')?.value||'active'
    };
    const r=await apiFetch(`/kb/documents/${id}/metadata`,{method:'PUT',body:JSON.stringify(payload)});
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Failed to update document metadata.');
    toast('Document metadata updated.');
    loadDocuments(); loadAnalytics();
    await editDocumentMeta(id);
  }catch(e){toast(e.message,true);}
}

async function viewDoc(id,name,offset=0) {
  revokeDocObjectUrl();
  docContentState={id,offset,limit:25,total:0,name:name||'Document Content'};
  document.getElementById('doc-modal').classList.add('show');
  document.getElementById('doc-modal-title').textContent=name||'Document Content';
  document.getElementById('doc-modal-sub').textContent='Original uploaded PDF/Word file';
  document.getElementById('doc-modal-body').innerHTML='<div class="loading">Loading original uploaded file...</div>';
  document.getElementById('doc-modal-foot').innerHTML='';
  await loadDocOriginal();
}

async function loadDocOriginal() {
  revokeDocObjectUrl();
  const state=docContentState;
  const body=document.getElementById('doc-modal-body');
  const foot=document.getElementById('doc-modal-foot');
  body.innerHTML='<div class="loading">Loading original uploaded file...</div>';
  foot.innerHTML='';
  try{
    const metaRes=await apiFetch(`/kb/documents/${state.id}`);
    const doc=await metaRes.json();
    if(!metaRes.ok) throw new Error(doc.detail||'Unable to load document details.');
    const fileRes=await fetch(`${API}/kb/documents/${state.id}/file`,{
      headers:{'Authorization':`Bearer ${token}`,'ngrok-skip-browser-warning':'true'}
    });
    if(fileRes.status===401){
      token=null; currentUser=null; clearAdminSession();
      showLogin('Your admin session expired. Please sign in again.');
      return;
    }
    if(!fileRes.ok){
      const err=await fileRes.json().catch(()=>({detail:'Unable to load original uploaded file.'}));
      throw new Error(err.detail||'Unable to load original uploaded file.');
    }
    const blob=await fileRes.blob();
    docObjectUrl=URL.createObjectURL(blob);
    const docCategory=doc.category||KB_CATEGORIES[0];
    const docTitle=doc.document_title||doc.original_filename||state.name||'Original document';
    const rawSourceUrl=String(doc.source_url||'').trim();
    const sourceUrl=safeUrl(doc.source_url);
    const sourceLine=sourceUrl
      ? `<a class="source-url-text" href="${esc(sourceUrl)}" target="_blank" rel="noopener noreferrer">${esc(rawSourceUrl)}</a>`
      : 'No source link saved';
    const fileType=String(doc.file_type||'').toLowerCase();
    const lowerName=String(doc.original_filename||'').toLowerCase();
    const isPdf=fileType==='pdf' || blob.type==='application/pdf' || lowerName.endsWith('.pdf');
    const isImage=['png','jpg','jpeg','tif','tiff','bmp','webp'].includes(fileType) || blob.type.startsWith('image/') || /\.(png|jpe?g|tiff?|bmp|webp)$/.test(lowerName);
    const fileKind=isPdf?'PDF':isImage?'scanned image':'Word document';
    const fileName=doc.original_filename||state.name||'document';
    const fileHref=esc(docObjectUrl);
    const downloadName=esc(fileName);
    document.getElementById('doc-modal-sub').textContent=`${docCategory} | Original ${fileKind} | status: ${(doc.status||'').toUpperCase()}`;
    body.innerHTML=`
      <div class="original-view-note">
        <span class="view-mode-pill">Original File</span>
        <div><strong>Viewing the exact uploaded ${fileKind}, not extracted text and not vector chunks.</strong><br/>This is the unedited file saved in the reference documents.</div>
      </div>
      <div class="doc-meta-grid">
        <div class="mini-metric"><div class="mini-metric-lbl">Status</div><div class="mini-metric-val" style="font-size:18px;">${esc((doc.status||'').toUpperCase())}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Original File</div><div class="mini-metric-val" style="font-size:18px;">${esc(fileKind)}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Size</div><div class="mini-metric-val" style="font-size:18px;">${doc.file_size?(doc.file_size/1048576).toFixed(2)+' MB':'-'}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Category</div><div class="mini-metric-val" style="font-size:18px;">${esc(docCategory)}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Original Link</div><div class="mini-metric-val" style="font-size:13px;line-height:1.35;">${sourceLine}</div></div>
      </div>
      <div class="full-doc-card">
        <div class="full-doc-toolbar">${documentTitleHtml(docTitle)}<span title="${esc(fileName)}">${esc(shortDocumentLabel(fileName))}</span></div>
        ${isPdf
          ? `<iframe class="original-file-frame" src="${fileHref}" title="${esc(docTitle)}"></iframe>`
          : isImage
            ? `<img class="original-image-preview" src="${fileHref}" alt="${esc(docTitle)}"/>`
            : `<div class="original-file-card">
              <span class="view-mode-pill">DOCX</span>
              <div><strong>Word files cannot be rendered faithfully inside the browser.</strong><br/>Use the button below to open or download the exact original Word document.</div>
              <div class="original-file-actions">
                <a class="btn btn-primary" href="${fileHref}" download="${downloadName}">Open / Download Original Word File</a>
              </div>
            </div>`}
      </div>`;
    foot.innerHTML=`<span style="font-size:12px;color:var(--muted);align-self:center;">Showing the exact uploaded file. Text preview and chunks are generated views.</span>
      <a class="btn btn-sm btn-gold" href="${fileHref}" download="${downloadName}">Download original</a>
      <button class="btn btn-sm btn-primary" onclick="loadDocTextPreview()">Text preview</button>
      <button class="btn btn-sm btn-primary" ${(doc.chunk_count??0)>0?'':'disabled'} onclick="loadDocContentPage(0)">View Text Sections</button>`;
  }catch(e){
    body.innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
    foot.innerHTML='<span style="font-size:12px;color:var(--muted);align-self:center;">The original uploaded file may be missing from disk.</span>';
  }
}

async function loadDocTextPreview() {
  revokeDocObjectUrl();
  const state=docContentState;
  const body=document.getElementById('doc-modal-body');
  const foot=document.getElementById('doc-modal-foot');
  body.innerHTML='<div class="loading">Loading readable text preview...</div>';
  foot.innerHTML='';
  try{
    const r=await apiFetch(`/kb/documents/${state.id}/original-content`);
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Unable to load readable text preview.');
    const doc=data.document||{};
    const docCategory=doc.category||KB_CATEGORIES[0];
    const text=data.text||'';
    const docTitle=doc.document_title||doc.original_filename||state.name||'Readable text preview';
    const rawSourceUrl=String(doc.source_url||'').trim();
    const sourceUrl=safeUrl(doc.source_url);
    const sourceLine=sourceUrl
      ? `<a class="source-url-text" href="${esc(sourceUrl)}" target="_blank" rel="noopener noreferrer">${esc(rawSourceUrl)}</a>`
      : 'No source link saved';
    const pageLabel=data.format==='pdf'
      ? `${data.page_count||0} pages`
      : `${text.split(/\n\n+/).filter(Boolean).length} sections`;
    document.getElementById('doc-modal-sub').textContent=`${docCategory} | Text preview from ${String(data.format||doc.file_type||'document').toUpperCase()} | status: ${(doc.status||'').toUpperCase()}`;
    body.innerHTML=`
      <div class="original-view-note">
        <span class="view-mode-pill">Text Preview</span>
        <div><strong>This is extracted readable text, not the original file layout.</strong><br/>Go back to Original File to view/download the unedited PDF/DOCX.</div>
      </div>
      <div class="doc-meta-grid">
        <div class="mini-metric"><div class="mini-metric-lbl">Status</div><div class="mini-metric-val" style="font-size:18px;">${esc((doc.status||'').toUpperCase())}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Text Preview</div><div class="mini-metric-val" style="font-size:18px;">${esc(pageLabel)}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Size</div><div class="mini-metric-val" style="font-size:18px;">${doc.file_size?(doc.file_size/1048576).toFixed(2)+' MB':'-'}</div></div>
        <div class="mini-metric"><div class="mini-metric-lbl">Original Link</div><div class="mini-metric-val" style="font-size:13px;line-height:1.35;">${sourceLine}</div></div>
      </div>
      <div class="full-doc-card">
        <div class="full-doc-toolbar">${documentTitleHtml(docTitle)}<span>${text.length.toLocaleString()} characters</span></div>
        <pre class="full-doc-text">${esc(text)}</pre>
      </div>`;
    foot.innerHTML=`<button class="btn btn-sm btn-primary" onclick="loadDocOriginal()">Back to Original File</button>
      <button class="btn btn-sm btn-primary" ${(doc.chunk_count??0)>0?'':'disabled'} onclick="loadDocContentPage(0)">View Text Sections</button>`;
  }catch(e){
    body.innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
    foot.innerHTML='<button class="btn btn-sm btn-primary" onclick="loadDocOriginal()">Back to Original File</button>';
  }
}

async function loadDocContentPage(offset=0) {
  revokeDocObjectUrl();
  const state=docContentState;
  state.offset=Math.max(offset,0);
  const body=document.getElementById('doc-modal-body');
  const foot=document.getElementById('doc-modal-foot');
  try{
    const r=await apiFetch(`/kb/documents/${state.id}/content?limit=${state.limit}&offset=${state.offset}`);
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Unable to load document content.');
    state.total=data.total_chunks||0;
    const doc=data.document||{};
    const docCategory=doc.category||KB_CATEGORIES[0];
    document.getElementById('doc-modal-sub').textContent=`${docCategory} | ${state.total} indexed chunks | status: ${(doc.status||'').toUpperCase()}`;
    if(!data.chunks?.length){
      body.innerHTML='<div class="empty">No indexed content is available for this document yet.</div>';
    }else{
      body.innerHTML=`
        <div class="doc-meta-grid">
          <div class="mini-metric"><div class="mini-metric-lbl">Status</div><div class="mini-metric-val" style="font-size:18px;">${esc((doc.status||'').toUpperCase())}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Chunks</div><div class="mini-metric-val" style="font-size:18px;">${state.total}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Size</div><div class="mini-metric-val" style="font-size:18px;">${doc.file_size?(doc.file_size/1048576).toFixed(2)+' MB':'-'}</div></div>
          <div class="mini-metric"><div class="mini-metric-lbl">Category</div><div class="mini-metric-val" style="font-size:18px;">${esc(docCategory)}</div></div>
        </div>
        ${data.chunks.map((c,idx)=>`<div class="chunk-card">
          <div class="chunk-head"><span>Chunk ${state.offset+idx+1}</span><span>${esc(c.metadata?.category||docCategory)} | ${c.metadata?.page?`Page ${esc(c.metadata.page)}`:'Document text'}</span></div>
          <div class="chunk-text">${esc(c.text)}</div>
        </div>`).join('')}`;
    }
    const prevDisabled=state.offset<=0?'disabled':'';
    const nextDisabled=state.offset+state.limit>=state.total?'disabled':'';
    foot.innerHTML=`<button class="btn btn-sm btn-primary" onclick="loadDocOriginal()">Back to Original File</button>
      <button class="btn btn-sm btn-primary" onclick="loadDocTextPreview()">Text preview</button>
      <button class="btn btn-sm btn-primary" ${prevDisabled} onclick="loadDocContentPage(${Math.max(0,state.offset-state.limit)})">Previous</button>
      <span style="font-size:12px;color:var(--muted);align-self:center;">Showing ${state.total?state.offset+1:0}-${Math.min(state.offset+state.limit,state.total)} of ${state.total}</span>
      <button class="btn btn-sm btn-primary" ${nextDisabled} onclick="loadDocContentPage(${state.offset+state.limit})">Next</button>`;
  }catch(e){
    body.innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
    foot.innerHTML='';
  }
}

function closeDocModal(e) {
  if(e && e.target && e.target.id!=='doc-modal') return;
  revokeDocObjectUrl();
  const modal=document.getElementById('doc-modal');
  modal.classList.remove('show','review-drawer');
  const close=modal.querySelector('.modal-hdr button');
  if(close){close.className='btn btn-sm btn-danger';close.textContent='Close';close.setAttribute('aria-label','Close');close.setAttribute('onclick','closeDocModal()');}
}
