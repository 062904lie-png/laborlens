/* Central review module; sessions stay in the existing backend. */
window.AdminReviews=(()=>{
 const U=AdminUI,$=id=>document.getElementById(id),api=(...a)=>AdminPages.api(...a);
 const sourceLabels=new Set(['High confidence match','Source match','Partial source match','Insufficient legal source','Conflicting sources','Temporary response issue']);
 const unresolvedLabels=new Set(['Insufficient legal source','Conflicting sources','Temporary response issue']);
 let active='needs_review',offset=0,version=0,detailVersion=0,summary={};

 function sourceMatchLabel(row={}){
  if(row.intent==='needs_clarification')return 'Awaiting user response';
  const level=String(row.confidence_level||'').trim().toLowerCase().replaceAll('_',' ');
  if(['provider unavailable','temporary response issue'].includes(level))return 'Temporary response issue';
  if(['conflicting','conflicting sources'].includes(level))return 'Conflicting sources';
  const stored=String(row.source_match_label||row.confidence_label||'').trim();
  if(sourceLabels.has(stored))return stored;
  if(['high','high confidence','high confidence match'].includes(level))return 'High confidence match';
  if(['direct','source','source match'].includes(level))return 'Source match';
  if(['partial','partial source match'].includes(level))return 'Partial source match';
  return 'Insufficient legal source';
 }
 function queryStatusKey(row={}){
  const saved=String(row.query_status||'').trim().toLowerCase();
  return ['resolved','unresolved','awaiting_user'].includes(saved)?saved:(unresolvedLabels.has(sourceMatchLabel(row))?'unresolved':'resolved');
 }
 function sourceMatchBadge(row){
  const label=sourceMatchLabel(row);
  const color=label==='High confidence match'?'green':label==='Source match'?'blue':label==='Partial source match'?'amber':label==='Conflicting sources'?'amber':'red';
  return U.badge(label,color);
 }
 function queryStatusBadge(row){if(queryStatusKey(row)==='awaiting_user')return U.badge('Awaiting user response','blue');return queryStatusKey(row)==='resolved'?U.badge('Resolved Query','green'):U.badge('Unresolved Query','red');}
 const feedback=row=>num(row.report_count)?U.badge('Reported','red'):num(row.not_helpful_count)?U.badge('Not Helpful','red'):num(row.helpful_count)?U.badge('Helpful','green'):'—';

 const page=document.createElement('section');page.id='tab-reviews';page.className='workbench-tab review-queue-page';
 page.innerHTML=`<div class="page-hdr"><div class="page-hdr-left"><h1>Review Queue</h1><p>Review chatbot answers, user feedback, reports, and conversations to ensure the information is accurate and reliable.</p></div></div><section class="review-summary-cards" id="review-summary-cards" aria-label="Review queue summary"><div class="loading">Loading review summary...</div></section><form class="workbench-filters review-filter-panel" id="review-filters"><label class="review-search-field">Search<input id="review-search" type="search" maxlength="200" placeholder="Question, user ID, or keywords..."/></label><label>Category<select id="review-category"><option value="">All categories</option>${KB_CATEGORIES.map(c=>`<option>${esc(c)}</option>`).join('')}</select></label><label>Language<select id="review-language"><option value="">All languages</option><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label><label>Feedback<select id="review-feedback"><option value="">All feedback</option><option value="yes">Helpful</option><option value="no">Not Helpful</option></select></label><label>Issue<select id="review-issue"><option value="">All issues</option><option value="reported">Reported</option><option value="not_helpful">Not Helpful</option><option value="unresolved">Unresolved</option><option value="other_needed">Other Review Needed</option></select></label><select id="review-query-status" hidden><option value=""></option><option value="resolved">Resolved</option><option value="unresolved">Unresolved</option></select><label>From<input type="date" id="review-from"/></label><label>To<input type="date" id="review-to"/></label><div class="review-filter-actions"><button class="btn btn-primary" type="submit">${U.icon('search')} Search / Filter</button><button class="btn" type="reset">Reset</button></div></form><section class="review-queue-table-card"><div id="review-rows"></div><footer class="review-queue-pager" id="review-pager"></footer></section>`;
 document.querySelector('.content-inner').append(page);
 $('review-filters').onsubmit=e=>{e.preventDefault();offset=0;load();};
 $('review-filters').onreset=()=>{active='needs_review';offset=0;setTimeout(load,0);};

 function metricValue(value){return Number.isFinite(Number(value))?Number(value).toLocaleString():'Unavailable';}
 function issueBadge(row){
  if(num(row.correction_closed))return U.badge('Completed','green');
  if(num(row.report_count))return U.badge('Reported','red');
  if(num(row.not_helpful_count))return U.badge('Not Helpful','red');
  if(queryStatusKey(row)==='unresolved')return U.badge('Unresolved','gold');
  if(queryStatusKey(row)==='awaiting_user')return U.badge('Awaiting user response','blue');
  const status=String(row.admin_review_status||'needs_review').replaceAll('_',' ');
  return U.badge(status.replace(/\b\w/g,letter=>letter.toUpperCase()),status==='completed'||status==='correct'||status==='resolved'?'green':'red');
 }
 function renderSummary(s,conversations){
  const metrics=[['chat','Needs Review',metricValue(s?.needs_review),'Answers currently requiring attention','red'],['check','Reviewed',metricValue(s?.completed),'Completed admin assessments','green'],['flag','Flagged / Reported',metricValue(s?.reported),'Reports submitted for review','gold'],['users','Total Conversations',conversations?metricValue(conversations.total):'Unavailable',conversations?'Recorded chat sessions':'Conversation count unavailable','blue']];
  $('review-summary-cards').innerHTML=metrics.map(([icon,label,value,note,color])=>`<article class="review-summary-card ${color}"><span>${U.icon(icon)}</span><div><small>${esc(label)}</small><strong>${esc(value)}</strong><p>${esc(note)}</p></div></article>`).join('');
 }
 function renderPager(total,limit){
  const pages=Math.max(1,Math.ceil(total/limit)),current=Math.min(Math.floor(offset/limit),pages-1);
  const start=Math.max(0,Math.min(current-2,pages-5)),end=Math.min(pages,start+5);
  const pageButtons=Array.from({length:end-start},(_,i)=>{const pageNumber=start+i;return `<button type="button" class="${pageNumber===current?'active':''}" data-review-offset="${pageNumber*limit}" aria-current="${pageNumber===current?'page':'false'}">${pageNumber+1}</button>`;}).join('');
  $('review-pager').innerHTML=`<span>Showing ${total?offset+1:0}&ndash;${Math.min(offset+limit,total)} of ${total}</span><nav aria-label="Review queue pagination"><button type="button" data-review-offset="${Math.max(0,offset-limit)}" ${current===0?'disabled':''}>‹ Previous</button>${pageButtons}<button type="button" data-review-offset="${Math.min((pages-1)*limit,offset+limit)}" ${current>=pages-1?'disabled':''}>Next ›</button></nav><label>Rows per page:<select id="review-page-size">${[7,10,20,50].map(size=>`<option value="${size}" ${limit===size?'selected':''}>${size}</option>`).join('')}</select></label>`;
  $('review-pager').querySelectorAll('[data-review-offset]').forEach(button=>button.onclick=()=>{if(button.disabled)return;offset=Number(button.dataset.reviewOffset);load();});
  $('review-page-size').onchange=event=>{localStorage.setItem('laborlens_admin_page_size',event.target.value);offset=0;load();};
 }
 async function load(){
  const v=++version,limit=Number($('review-page-size')?.value||adminPageSize());$('review-rows').innerHTML='<div class="loading">Loading review queue...</div>';
  const p=new URLSearchParams({q:$('review-search').value,category:$('review-category').value,language:$('review-language').value,feedback:$('review-feedback').value,query_status:$('review-query-status').value,offset,limit});
  if(active==='needs_review')p.set('needs_attention','true');
  else if(active==='completed')p.set('completed','true');
  const issue=$('review-issue').value;
  if(issue==='reported')p.set('reports_only','true');
  if(issue==='not_helpful')p.set('feedback','no');
  if(issue==='unresolved')p.set('query_status','unresolved');
  if(issue==='other_needed')p.set('other_issue','true');
  if($('review-from').value)p.set('date_from',$('review-from').value);
  if($('review-to').value)p.set('date_to',$('review-to').value);
  try{
   const [data,s,conversations]=await Promise.all([api('/admin/answer-reviews?'+p),api('/admin/review-summary'),api('/admin/conversations?limit=1').catch(()=>null)]);
   if(v!==version)return;summary=s;
   if(offset>=data.total&&offset){offset=0;return load();}
   renderSummary(s,conversations);
   if(!data.items.length){$('review-rows').innerHTML='<div class="review-empty"><strong>No answers match your current filters.</strong><button class="btn" type="button" id="review-reset-empty">Reset Filters</button></div>';$('review-reset-empty').onclick=()=>$('review-filters').reset();$('review-pager').innerHTML='';return;}
   $('review-rows').innerHTML=U.table(['Question','Topic','Language','Issue','Date','Action'],data.items.map(row=>[esc(row.question||'Not recorded'),esc(row.category||'Others'),esc(languageLabel(row.language)),issueBadge(row),esc(dateLabel(row.created_at)),`<button class="btn btn-sm btn-primary" data-review="${Number(row.id)}">Review</button>`]),'review-mobile-cards');
   $('review-rows').querySelectorAll('[data-review]').forEach(button=>button.onclick=()=>open(Number(button.dataset.review)));
   renderPager(data.total,limit);
  }catch(error){if(v===version){$('review-summary-cards').innerHTML='';$('review-rows').innerHTML='<div class="err-msg">Review queue could not be loaded. <button class="btn" type="button" id="retry-reviews">Retry</button></div>';$('retry-reviews').onclick=load;$('review-pager').innerHTML='';}}
 }

 function setFilter(filter='all'){
  active=filter==='resolved'?'completed':['needs_review','completed'].includes(filter)?filter:'all';offset=0;
  $('review-issue').value=['reported','not_helpful','unresolved'].includes(filter)?filter:'';
  if(['reported','not_helpful','unresolved','needs_review'].includes(filter)){ $('review-filters').reset();$('review-issue').value=['reported','not_helpful','unresolved'].includes(filter)?filter:'';active=filter==='needs_review'?'needs_review':'all';}
  $('review-query-status').value='';$('review-feedback').value='';
  return load();
 }
 function contextSources(value){let rows=value;if(typeof rows==='string'){try{rows=JSON.parse(rows);}catch(error){rows=[];}}if(!Array.isArray(rows)||!rows.length)return '';return '<details><summary>Stored sources ('+rows.length+')</summary>'+rows.map(source=>'<p><strong>'+documentTitleHtml(source.document_title||source.source||source.filename||'Source')+'</strong><br>'+esc(source.text||source.content||source.excerpt||'Excerpt not recorded')+'</p>').join('')+'</details>';}
 function contextHTML(rows,current){return rows.length?rows.map(row=>`<article class="context-message ${row.role==='assistant'?'assistant':''} ${row.id===current?'current':''}"><strong>${esc(row.role==='assistant'?'LaborLens':'User')}</strong><small>${esc(row.created_at||'')}</small><p>${esc(row.content)}</p>${row.feedback?U.badge(row.feedback==='yes'?'Helpful':'Not Helpful'):''}${contextSources(row.sources)}${row.role==='assistant'?`<button class="btn btn-sm" data-context-review="${Number(row.id)}">Review answer</button>`:''}</article>`).join(''):'<div class="empty">No context recorded.</div>';}
 function bindContext(element){element.querySelectorAll('[data-context-review]').forEach(button=>button.onclick=()=>open(Number(button.dataset.contextReview)));}
 function reviewStatusBadge(row){
  if(num(row.correction_closed))return U.badge('Completed','green');
  if(num(row.report_count))return U.badge('Reported','red');
  if(num(row.not_helpful_count))return U.badge('Not Helpful','red');
  if(queryStatusKey(row)==='unresolved')return U.badge('Unresolved','gold');
  if(queryStatusKey(row)==='awaiting_user')return U.badge('Awaiting user response','blue');
  const status=String(row.admin_review_status||'needs_review').replaceAll('_',' ');
  return U.badge(status.replace(/\b\w/g,letter=>letter.toUpperCase()),['correct','completed','resolved'].includes(status)?'green':'red');
 }
 function dateAsked(value){
  if(!value)return ['Unavailable',''];
  const parsed=new Date(value);if(Number.isNaN(parsed.getTime()))return [String(value).slice(0,10),''];
  const date=parsed.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});
  const hasTime=/[T\s]\d{1,2}:\d{2}/.test(String(value));
  return [date,hasTime?parsed.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}):''];
 }
 async function copyReviewText(value){
  try{await navigator.clipboard.writeText(value);toast('Copied to clipboard.');}
  catch(error){const field=document.createElement('textarea');field.value=value;document.body.append(field);field.select();document.execCommand('copy');field.remove();toast('Copied to clipboard.');}
 }

 async function open(id){
  const v=++detailVersion,opener=document.activeElement;U.drawer('Review Details','<div class="loading">Loading review...</div>');
  const modal=$('doc-modal'),closeButton=modal.querySelector('.modal-hdr button');
  $('doc-modal-sub').textContent='Review this answer, check the sources, and take the appropriate action.';
  closeButton.className='review-drawer-close';closeButton.textContent='×';closeButton.setAttribute('aria-label','Close review details');
  const closeReview=()=>{closeDocModal();opener?.focus();};closeButton.onclick=closeReview;
  try{
   const row=await api(`/admin/answer-reviews/${id}`);
   if(v!==detailVersion||!$('doc-modal').classList.contains('show'))return;
   const sources=Array.isArray(row.sources)?row.sources:[];
   const match=sourceMatchLabel(row),queryStatus=queryStatusKey(row);
   const [askedDate,askedTime]=dateAsked(row.created_at);
   const answer=`<label class="form-group">Answer Rating<select id="answer-assessment"><option value="">Choose your rating</option><option value="correct">Correct</option><option value="partial_correct">Partly Correct</option><option value="incorrect">Incorrect</option><option value="kb_update">Needs Document Update</option></select></label><label class="form-group">Review Note<textarea id="answer-note" maxlength="2000" placeholder="Add a note about this review...">${esc(row.admin_review_note||'')}</textarea></label><p class="review-helper">Your rating is saved. The original answer stays the same.</p>`;
   const sourceHTML=sources.length?sources.map(source=>{const url=safeUrl(source.source_url||source.url);return `<article class="review-source-card"><div><strong>${documentTitleHtml(source.document_title||source.source||source.filename||'Retrieved source')}</strong>${source.article_section||source.article?`<small>${esc(source.article_section||source.article)}</small>`:''}</div><details><summary>Read source excerpt</summary><p>${esc(source.text||source.content||source.excerpt||'Source excerpt not stored.')}</p></details>${url?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open legal source</a>`:''}</article>`;}).join(''):`<article class="review-source-warning"><strong>⚠ Insufficient legal source</strong><span>No sources recorded for this answer.</span></article>`;
   const feedbackItems=(row.feedback||[]);
   const feedbackHTML=feedbackItems.length?feedbackItems.map(item=>`<article class="review-feedback-item">${U.badge(item.rating==='yes'?'Helpful':'Not Helpful',item.rating==='yes'?'green':'red')}<p>${esc(item.comment||'No comment provided')}</p><small>${esc(dateAsked(item.created_at).filter(Boolean).join(' · ')||'Date not recorded')}</small></article>`).join(''):'<article class="review-feedback-item"><p>No feedback or reports recorded.</p></article>';
   const workflowHTML=feedbackItems.length?feedbackItems.map(item=>`<article class="review-workflow-item"><label class="form-group">Report status<select id="report-${Number(item.id)}">${['pending','under_review','resolved','dismissed','reviewed'].map(status=>`<option value="${status}" ${status===item.review_status?'selected':''}>${status.replaceAll('_',' ')}</option>`).join('')}</select></label><label class="form-group">Report note<textarea id="report-note-${Number(item.id)}" maxlength="2000" placeholder="Add a note about this review...">${esc(item.admin_notes||'')}</textarea></label><button class="btn btn-sm" data-report="${Number(item.id)}">Save report status</button></article>`).join(''):'<article class="review-workflow-item"><p>No report status is recorded.</p></article>';
   $('doc-modal-body').innerHTML=`<section class="review-detail-summary" aria-label="Review document details"><article><small>Status</small>${reviewStatusBadge(row)}</article><article><small>Date Asked</small><strong>${esc(askedDate)}</strong>${askedTime?`<span>${esc(askedTime)}</span>`:''}</article></section><section class="review-section"><h3>User Question</h3><article class="review-question-card"><button class="btn btn-sm review-copy" type="button" data-copy-question>▣ Copy</button><p>${esc(row.question||'Question not recorded')}</p><small>${esc(languageLabel(row.language))}</small></article></section><section class="review-section"><h3>LaborLens Answer</h3><article class="review-answer"><p>${esc(row.answer||'Answer not recorded')}</p><button class="btn btn-sm review-copy" type="button" data-copy-answer>▣ Copy</button></article></section><section class="review-section"><h3>Sources Used</h3>${sourceHTML}</section>${feedbackItems.length?`<details class="review-accordion"><summary>Feedback and reports (${feedbackItems.length})</summary>${feedbackHTML}<details><summary>Manage report status</summary>${workflowHTML}</details></details>`:''}<details class="review-accordion review-context"><summary>Related Messages</summary><button class="btn btn-sm" id="full-context" type="button">▣ View Full Chat</button><div id="nearby-context" class="loading">Loading nearby messages...</div><div id="full-context-output"></div></details><section class="review-admin-section"><h3>Admin Review</h3>${answer}</section>`;


   $('answer-assessment').value=['correct','partial_correct','incorrect','kb_update'].includes(row.admin_review_status)?row.admin_review_status:'';
   $('doc-modal-foot').innerHTML='<button class="btn" id="open-correction">▣ Create / Open Correction</button><div><button class="btn" id="cancel-review">Cancel</button><button class="btn btn-primary" id="save-assessment">Save Review</button></div>';
   $('open-correction').onclick=()=>AdminMaintenance.correction(id);
   $('cancel-review').onclick=closeReview;
   $('doc-modal-body').querySelector('[data-copy-question]').onclick=()=>copyReviewText(row.question||'');
   $('doc-modal-body').querySelector('[data-copy-answer]').onclick=()=>copyReviewText(row.answer||'');

   $('save-assessment').onclick=async()=>{const button=$('save-assessment'),status=$('answer-assessment').value;if(!status)return toast('Choose your rating.',true);button.disabled=true;try{await api(`/admin/answer-reviews/${id}`,{method:'PUT',body:JSON.stringify({status,note:$('answer-note').value})});toast(status==='kb_update'?'Review saved. Create a correction to track the KB update.':'Review saved.');await open(id);load();}catch(error){toast(error.message,true);}finally{button.disabled=false;}};
   $('doc-modal-body').querySelectorAll('[data-report]').forEach(button=>button.onclick=async()=>{button.disabled=true;try{const feedbackId=button.dataset.report;await api(`/admin/review-queue/${feedbackId}`,{method:'PUT',body:JSON.stringify({review_status:$('report-'+feedbackId).value,admin_notes:$('report-note-'+feedbackId).value})});toast('Report workflow saved.');load();}catch(error){toast(error.message,true);}finally{button.disabled=false;}});
   $('full-context').onclick=()=>fullContext(row.session_id,0,v);
   const contextPanel=$('nearby-context').closest('details');let contextLoaded=false;
   contextPanel.ontoggle=async()=>{if(!contextPanel.open||contextLoaded)return;contextLoaded=true;try{const context=await api(`/admin/answer-reviews/${id}/context`);if(v!==detailVersion)return;$('nearby-context').className='';$('nearby-context').innerHTML=contextHTML(context.items,id);bindContext($('nearby-context'));}catch(error){if(v===detailVersion)$('nearby-context').textContent='Unable to load nearby messages. Try reopening this review.';}};
  }catch(error){if(v===detailVersion)$('doc-modal-body').innerHTML='<div class="err-msg">Unable to load review details.</div>';}
 }
 async function fullContext(session,start,v){const host=$('full-context-output');host.innerHTML='<div class="loading">Loading conversation context...</div>';try{const data=await api(`/admin/conversation-review/${encodeURIComponent(session)}?offset=${start}&limit=25`);if(v!==detailVersion)return;host.innerHTML=contextHTML(data.items)+`<div class="workbench-pager" id="context-pager"></div>`;bindContext(host);U.pager($('context-pager'),start,data.total,next=>fullContext(session,next,v));}catch(error){if(v===detailVersion)host.textContent='Unable to load full context.';}}
 async function filterTopic(category,filters={}){$('review-filters').reset();active='all';offset=0;$('review-category').value=category;$('review-language').value=filters.language||'';$('review-from').value=filters.date_from||'';$('review-to').value=filters.date_to||'';return load();}
 return {load,setFilter,open,filterTopic,setSearch(value){$('review-search').value=value;offset=0;active='all';}};
})();
