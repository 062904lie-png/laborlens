/* Admin-only review workflows. No chatbot answer generation or KB auto-corrections. */
(() => {
  const $ = id => document.getElementById(id);
  const states = {reviews:{offset:0,version:0},reports:{offset:0,version:0}};
  let documentOffset=0, documentSignature='', detailVersion=0;
  const option = (value,label=value) => `<option value="${esc(value)}">${esc(label.replace(/_/g,' '))}</option>`;
  const statusBadge = value => `<span class="badge ${['correct','resolved','Ready','Connected'].includes(value)?'badge-green':['incorrect','Unavailable'].includes(value)?'badge-red':'badge-amber'}">${esc(String(value||'unreviewed').replace(/_/g,' '))}</span>`;
  const table = (headers, rows, cls='') => `<div class="tbl-wrap"><table class="workbench-table ${cls}"><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${headers.length}" class="empty">No data available.</td></tr>`}</tbody></table></div>`;
  const card = (label,value) => {const labels={'Total Users':['blue','♙'],'Conversations':['green','☏'],'KB Documents':['gold','▤'],'Stored KB Chunks':['purple','▥'],'Questions Today (UTC)':['red','?'],'Average Response Time':['blue','◷'],'Helpful Rate':['green','✓'],'Pending Reports':['red','⚑']};const [color,icon]=labels[label]||['blue','•'];return `<div class="stat-card ${color}"><span class="stat-icon" aria-hidden="true">${icon}</span><div class="stat-lbl">${esc(label)}</div><div class="stat-val">${esc(value ?? '—')}</div></div>`;};
  async function request(path, opts={}) {
    const r=await apiFetch(path,opts);
    if(!r.ok) throw new Error(r.status===401?'Session expired. Please sign in.':r.status===403?'Administrator access required.':'Unable to complete this request. Check the backend and try again.');
    return r.json();
  }
  function refreshVisibleReviews(){for(const kind of Object.keys(states)){if($(`tab-${kind}`).classList.contains('tab-active'))loadReviews(kind);}}
  function addPanel(id,title,description,html) {
    const el=document.createElement('section'); el.id=`tab-${id}`;el.className='workbench-tab';
    el.innerHTML=`<div class="page-hdr"><div class="page-hdr-left"><h1>${title}</h1><p>${description}</p></div></div>${html}`;
    document.querySelector('.content-inner').append(el);
    const nav=document.createElement('button');nav.type='button';nav.className='nav-item';nav.dataset.tab=id;
    nav.innerHTML=`<span class="nav-icon" aria-hidden="true">${id==='reports'?'⚑':'✓'}</span>${title}`;
    nav.addEventListener('click',()=>switchTab(nav,id));
    document.querySelector('#left-nav [data-tab="analytics"]').before(nav);
  }
  function pager(id,state,total,action) {
    const el=$(id);if(!el)return;
    el.innerHTML=`<span>${total?`${state.offset+1}–${Math.min(state.offset+25,total)} of ${total}`:'No results'}</span><div><button type="button" class="btn btn-sm btn-primary" data-direction="-1" ${state.offset===0?'disabled':''}>Previous</button><button type="button" class="btn btn-sm btn-primary" data-direction="1" ${state.offset+25>=total?'disabled':''}>Next</button></div>`;
    el.querySelectorAll('button').forEach(b=>b.onclick=()=>action(Number(b.dataset.direction)));
  }
  async function loadReviews(kind) {
    const state=states[kind], version=++state.version;
    const params=new URLSearchParams({offset:state.offset,limit:25,q:$(`${kind}-search`).value,language:$(`${kind}-language`).value});
    params.set('category',$(`${kind}-category`).value);
    params.set(kind==='reports'?'report_status':'review_status',$(`${kind}-status`).value);
    if(kind==='reports')params.set('reports_only','true');
    $(`${kind}-rows`).innerHTML='<div class="loading">Loading answers...</div>';
    try {
      const data=await request(`/admin/answer-reviews?${params}`);
      if(version!==state.version)return;
      if(state.offset>=data.total && state.offset>0){state.offset=0;return loadReviews(kind);}
      $(`${kind}-rows`).innerHTML=kind==='reports'?table(['Question','Category / Intent','Language','Report reason','Report status','Reported','Action'],data.items.map(r=>[esc(r.question),esc(r.category||r.intent||'Not recorded'),esc(languageLabel(r.language)),esc(r.report_reason),statusBadge(r.report_status),esc(dateLabel(r.reported_at)),`<button type="button" class="btn btn-sm btn-primary" data-review="${Number(r.id)}">View / Review</button>`])):table(['Question','Answer preview','Category / Intent','Language','Source match','Assessment','Reports','Date','Action'],data.items.map(r=>[
        esc(r.question||'Question not recorded'),esc(r.answer_preview),esc(r.category||r.intent||'Not recorded'),esc(languageLabel(r.language)),esc(r.confidence_level||'Not recorded'),statusBadge(r.admin_review_status),esc(r.report_count),esc(dateLabel(r.created_at)),`<button type="button" class="btn btn-sm btn-primary" data-review="${Number(r.id)}">View / Review</button>`]));
      $(`${kind}-rows`).querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>openReview(Number(b.dataset.review)));
      pager(`${kind}-pager`,state,data.total,direction=>{state.offset+=direction*25;loadReviews(kind);});
    } catch(e) {if(version===state.version)$(`${kind}-rows`).innerHTML=`<div class="err-msg" role="alert">${esc(e.message)}</div>`;}
  }
  async function openReview(id) {
    const version=++detailVersion;
    revokeDocObjectUrl(); $('doc-modal').classList.add('show','review-drawer');
    $('doc-modal-title').textContent=`Answer review #${id}`;
    $('doc-modal-sub').textContent='Administrative assessment — not legal representation';
    $('doc-modal-body').innerHTML='<div class="loading">Loading answer and sources...</div>';$('doc-modal-foot').innerHTML='';
    try {
      const row=await request(`/admin/answer-reviews/${id}`);
      if(version!==detailVersion || !$('doc-modal').classList.contains('show'))return;
      const sources=Array.isArray(row.sources)?row.sources:[];
      $('doc-modal-body').innerHTML=`<div class="workbench-detail"><h3>User question</h3><div class="review-answer">${esc(row.question||'Not recorded')}</div><h3>LaborLens answer</h3><div class="review-answer">${esc(row.answer)}</div>
      <p class="workbench-note">${esc(languageLabel(row.language))} · ${esc(row.intent||'No intent')} · ${esc(row.confidence_level||'No source match')}<br>${esc(row.confidence_reason||'')}<br>${esc(row.diagnostic_note)}</p>
      <h3>Sources / legal basis</h3>${sources.length?sources.map(s=>{const url=safeUrl(s.source_url||s.url);return `<div class="chunk-card"><strong>${esc(s.document_title||s.source||s.filename||'Retrieved source')}</strong><p>${esc(s.article_section||s.article||'')} ${s.page?`Page ${esc(s.page)}`:''}</p>${url?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open source</a>`:''}<p>${esc(s.text||s.content||s.excerpt||'Source text not stored in this record.')}</p></div>`;}).join(''):'<p>No sources stored for this answer.</p>'}
      <h3>User feedback and reports</h3>${row.feedback.length?row.feedback.map(f=>`<div class="chunk-card"><strong>${esc(f.rating==='yes'?'Helpful':'Not helpful')}</strong><p>${esc(f.comment||'No comment')}</p><p>Report/review status: ${esc(f.review_status)}</p><p>${esc(f.admin_notes||'')}</p><div class="form-group"><label for="report-${f.id}">Update feedback workflow</label><select id="report-${f.id}">${['pending','under_review','resolved','dismissed','reviewed'].map(v=>option(v)).join('')}</select></div><button class="btn btn-sm btn-gold" data-feedback="${Number(f.id)}">Save workflow status</button></div>`).join(''):'<p>No feedback for this answer.</p>'}
      <h3>Admin assessment</h3><div class="form-group"><label for="answer-assessment">Assessment</label><select id="answer-assessment">${['unreviewed','correct','needs_review','incorrect','resolved'].map(v=>option(v)).join('')}</select></div><div class="form-group"><label for="answer-note">Admin note (for example: needs KB update or prompt/logic fix)</label><textarea id="answer-note" maxlength="2000">${esc(row.admin_review_note||'')}</textarea></div><p class="workbench-note">Reviewed by admin ID ${esc(row.admin_reviewed_by??'—')} · ${esc(row.admin_reviewed_at||'Not reviewed')}<br>Saving a review does not alter the answer, sources, or knowledge base.</p></div>`;
      $('answer-assessment').value=row.admin_review_status||'unreviewed';
      row.feedback.forEach(f=>{$(`report-${f.id}`).value=f.review_status||'pending';});
      row.feedback.forEach(f=>{
        const field=document.createElement('div');field.className='form-group';
        field.innerHTML=`<label for="report-note-${Number(f.id)}">Report / feedback admin note</label><textarea id="report-note-${Number(f.id)}" maxlength="2000">${esc(f.admin_notes||'')}</textarea>`;
        $(`report-${f.id}`).closest('.chunk-card').querySelector('button').before(field);
      });
      $('doc-modal-body').querySelectorAll('[data-feedback]').forEach(button=>button.onclick=async()=>{
        const feedbackId=Number(button.dataset.feedback);
        const note=$(`report-note-${feedbackId}`).value;
        button.disabled=true;
        try{await request(`/admin/review-queue/${feedbackId}`,{method:'PUT',body:JSON.stringify({review_status:$(`report-${feedbackId}`).value,admin_notes:note})});toast('Feedback status saved.');refreshVisibleReviews();openReview(id);}catch(e){toast(e.message,true);}finally{button.disabled=false;}
      });
      $('doc-modal-foot').innerHTML='<button type="button" class="btn btn-primary" id="save-answer-review">Save Assessment</button><button type="button" class="btn btn-gold" id="view-review-conversation">View Conversation</button>';
      $('view-review-conversation').onclick=()=>openConversation(row.session_id);
      $('save-answer-review').onclick=async function(){
        this.disabled=true;
        try{await request(`/admin/answer-reviews/${id}`,{method:'PUT',body:JSON.stringify({status:$('answer-assessment').value,note:$('answer-note').value})});toast('Assessment saved.');refreshVisibleReviews();await openReview(id);}catch(e){toast(e.message,true);}finally{this.disabled=false;}
      };
    }catch(e){if(version===detailVersion)$('doc-modal-body').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;}
  }
  async function openConversation(session,offset=0) {
    $('doc-modal').classList.add('review-drawer');
    const version=++detailVersion;
    $('doc-modal-title').textContent='Conversation review';$('doc-modal-sub').textContent='Chronological saved messages — 25 per page';
    $('doc-modal-body').innerHTML='<div class="loading">Loading conversation...</div>';$('doc-modal-foot').innerHTML='';
    try{
      const data=await request(`/admin/conversation-review/${encodeURIComponent(session)}?offset=${offset}`);
      if(version!==detailVersion||!$('doc-modal').classList.contains('show'))return;
      $('doc-modal-body').innerHTML=data.items.map(m=>`<article class="chunk-card"><h3>${esc(m.role)} · ${esc(languageLabel(m.language))}</h3><p class="workbench-note">${esc(m.created_at)} · ${esc(m.confidence_level||'')}</p><div class="review-answer">${esc(m.content)}</div>${m.role==='assistant'?`<button class="btn btn-sm btn-primary" data-review="${Number(m.id)}">Sources / Feedback / Assessment</button>`:''}</article>`).join('')||'<div class="empty">No messages.</div>';
      $('doc-modal-body').querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>openReview(Number(b.dataset.review)));
      $('doc-modal-foot').innerHTML='<div id="conversation-pager" class="workbench-pager"></div>';
      pager('conversation-pager',{offset},data.total,d=>openConversation(session,offset+d*25));
    }catch(e){if(version===detailVersion)$('doc-modal-body').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;}
  }
  async function loadHealth(target='health-items') {
    const el=$(target);if(!el)return;
    try{
      const data=await request('/admin/system-health');
      el.innerHTML=`<p class="workbench-note">Checked ${esc(data.checked_at)}. ${esc(data.note)}</p><div class="health-components">${data.components.map(c=>`<div class="mini-metric"><div class="mini-metric-lbl">${esc(c.name)}</div>${statusBadge(c.status)}${c.chunks!=null?`<p>${esc(c.chunks)} chunks</p>`:''}</div>`).join('')}</div>`;
      el.innerHTML+=target==='dashboard-health'?`<div class="health-components">${data.providers.map(p=>`<div class="mini-metric"><div class="mini-metric-lbl">${esc(p.model)}</div>${statusBadge(p.status)}</div>`).join('')}</div>`:table(['Provider / Model','Status','Circuit','Requests','Successes','Failures','429','Timeouts','Mean latency','Last success','Last failure','Last error'],data.providers.map(p=>[`${esc(p.provider)}<br>${esc(p.model)}<br>${p.configured?'Configured ✓':'Not configured'}`,statusBadge(p.status),esc(p.circuit_state||'No observations'),...['requests','successes','failures','rate_limits','timeouts'].map(k=>esc(p[k]??0)),p.requests?`${esc(p.average_latency_ms)} ms`:'No observations',p.last_success_at?esc(new Date(p.last_success_at*1000).toISOString()):'Not recorded',p.last_failure_at?esc(new Date(p.last_failure_at*1000).toISOString()):'Not recorded',esc(p.last_error_type||'None recorded')]),'health-table');
    }catch(e){el.innerHTML=`<div class="err-msg">Provider health data is temporarily unavailable. ${esc(e.message)}</div>`;}
  }
  async function dashboard(data) {
    try{
      const summary=await request('/admin/evaluation-summary');
      const t=data.totals||{},f=data.feedback_summary||{};
      $('stat-grid').innerHTML=[['Total Users',t.users],['Conversations',t.sessions],['KB Documents',summary.documents.total],['Stored KB Chunks',summary.documents.stored_chunks],['Questions Today (UTC)',summary.usage.today],['Average Response Time',summary.performance.average==null?'—':`${Math.round(summary.performance.average)} ms`],['Helpful Rate',f.total?`${pct(f.helpful,f.total)}%`:'No ratings'],['Pending Reports',summary.reports.pending]].map(([label,value])=>card(label,value)).join('');
      $('evaluation-detail').innerHTML=`<p class="workbench-note">${esc(summary.note)}</p><div class="stat-grid">${[['Questions / 7 days',summary.usage.week],['Questions / 30 days',summary.usage.month],['Explicit reports',summary.reports.total],['Recorded timings',summary.performance.samples]].map(([k,v])=>card(k,v)).join('')}</div>`+table(['Timing metric','Milliseconds'],['average','median','p90','p95','fastest','slowest'].map(k=>[esc(k),summary.performance[k]==null?'No data available':esc(Math.round(summary.performance[k]))]))+table(['Admin assessment','Answers'],summary.reviews.map(r=>[statusBadge(r.status),esc(r.count)]))+table(['Source-match label (not accuracy)','Answers'],(data.source_match_distribution||[]).map(r=>[esc(r.source_match),esc(r.count)]));
    }catch(e){$('evaluation-detail').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;}
    await loadHealth('dashboard-health');
    try{
      const rows=await request('/admin/answer-reviews?reports_only=true&limit=5');
      $('dashboard-reports').innerHTML=table(['Question','Report reason','Report status','Date','Action'],rows.items.map(r=>[esc(r.question),esc(r.report_reason),statusBadge(r.report_status),esc(dateLabel(r.reported_at)),`<button class="btn btn-sm btn-primary" data-review="${Number(r.id)}">View</button>`]));
      $('dashboard-reports').querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>openReview(Number(b.dataset.review)));
    }catch(e){$('dashboard-reports').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;}
  }
  function documentPage(rows) {
    const signature=rows.map(r=>r.id).join(',');if(signature!==documentSignature){documentOffset=0;documentSignature=signature;}
    let footer=$('document-pager');if(!footer){footer=document.createElement('div');footer.id='document-pager';footer.className='workbench-pager';$('docs-tbody').closest('.panel').append(footer);}
    pager('document-pager',{offset:documentOffset},rows.length,d=>{documentOffset+=d*25;renderDocumentsTable();});
    if($('dashboard-uploads'))$('dashboard-uploads').innerHTML=table(['Recent document','Primary category','Legal status','Upload date','Uploaded by'],[...kbDocumentsCache].sort((a,b)=>new Date(b.created_at)-new Date(a.created_at)).slice(0,5).map(r=>[esc(documentDisplayTitle(r)),esc(r.primary_category||r.category),statusBadge(r.version_status),esc(dateLabel(r.created_at)),r.uploaded_by?`Admin ID ${esc(r.uploaded_by)}`:'Not recorded']));
    return rows.slice(documentOffset,documentOffset+25);
  }
  async function archive(id) {
    if(!confirm('Archive this document? It will remain stored for historical reference and be excluded from current chatbot retrieval.'))return;
    try{await request(`/kb/documents/${id}/metadata`,{method:'PUT',body:JSON.stringify({is_active:false,version_status:'archived'})});toast('Document archived; original file retained.');loadDocuments();}catch(e){toast(e.message,true);}
  }
  function onTab(name){if(states[name])loadReviews(name);if(name==='performance')loadPerformance();if(name==='analytics')loadOverview();window.AdminPages?.open(name);}
  function setup(){
    for(const kind of ['reviews','reports']){
      const statuses=kind==='reports'?['pending','under_review','resolved','dismissed','reviewed']:['unreviewed','correct','needs_review','incorrect','resolved'];
      addPanel(kind,kind==='reports'?'Reported Answers':'Chat Reviews',kind==='reports'?'Explicit “Report incorrect answer” feedback. Ordinary not-helpful ratings remain in User Feedback.':'Assess saved answers against their recorded sources. Source similarity is not an accuracy score.',`<form id="${kind}-filters" class="workbench-filters"><label>Search question or answer<input id="${kind}-search" maxlength="200" type="search"/></label><label>Language<select id="${kind}-language">${option('','All languages')}${['en','fil','hil'].map(v=>option(v,languageLabel(v))).join('')}</select></label><label>${kind==='reports'?'Report status':'Assessment'}<select id="${kind}-status">${option('','All statuses')}${statuses.map(v=>option(v)).join('')}</select></label><button class="btn btn-primary" type="submit">Search / Refresh</button></form><div class="panel"><div id="${kind}-rows"></div><div class="workbench-pager" id="${kind}-pager"></div></div>`);
      $(`${kind}-filters`).onsubmit=e=>{e.preventDefault();states[kind].offset=0;loadReviews(kind);};
      const category=document.createElement('label');category.innerHTML=`Primary category<select id="${kind}-category">${option('','All categories')}${KB_CATEGORIES.map(v=>option(v)).join('')}</select>`;
      $(`${kind}-filters`).querySelector('button').before(category);
    }
    const nav=$('left-nav'),scroll=document.createElement('div');scroll.className='admin-nav-scroll';
    const groups=[['Dashboard',['overview']],['Knowledge Management',['knowledge','faqs']],['Chat Quality',['reviews','reports','queries']],['Management',['users']],['Monitoring',['analytics','performance']]];
    for(const [title,tabs] of groups){const label=document.createElement('div');label.className='nav-section-lbl';label.textContent=title;scroll.append(label);tabs.forEach(id=>scroll.append(nav.querySelector(`[data-tab="${id}"]`)));}
    nav.replaceChildren(scroll);
    const footer=document.createElement('div');footer.className='admin-nav-footer';footer.innerHTML='<button class="end-btn" type="button">Sign Out</button>';footer.querySelector('button').onclick=()=>{detailVersion++;closeDocModal();closeAdminNav();doLogout();};nav.append(footer);
    ['health','reports','uploads'].forEach(key=>$('tab-overview').insertAdjacentHTML('beforeend',`<div class="panel"><div class="panel-hdr"><h2 class="panel-title">${{health:'System Health',reports:'Recent Reported Answers',uploads:'Recent KB Uploads'}[key]}</h2></div><div class="panel-body" id="dashboard-${key}"><div class="loading">Loading...</div></div></div>`));
    $('tab-analytics').insertAdjacentHTML('afterbegin','<div class="panel"><div class="panel-hdr"><h2 class="panel-title">Quality and Performance Evaluation</h2></div><div class="panel-body" id="evaluation-detail"></div></div>');
    const searchInput=$('file-input');searchInput.onchange=()=>{const file=searchInput.files[0];$('upload-msg').textContent=file?`Selected: ${file.name}`:'';};
    const uploadButton=document.createElement('button');uploadButton.type='button';uploadButton.className='btn btn-primary';uploadButton.textContent='Upload and Process';
    uploadButton.onclick=async()=>{if(!searchInput.files[0])return toast('Select a document first.',true);uploadButton.disabled=true;try{await uploadFile(searchInput);}finally{uploadButton.disabled=false;}};
    searchInput.after(uploadButton);
    $('kb-version-status').insertAdjacentHTML('beforeend',option('under_review','Under Review'));
  }
  window.AdminWorkbench={onTab,dashboard,loadHealth,documentPage,archive,openConversation,openReview};
  setup();
})();
