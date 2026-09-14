/* Maintenance tools use authenticated APIs and escape all stored/generated text. */
window.AdminMaintenance = (() => {
  const U = AdminUI;
  const stages = ['needs_kb_update','update_linked','ready_for_retest','closed'];
  const stageLabels={needs_kb_update:'Problem',update_linked:'Documents Updated',ready_for_retest:'Test Again',closed:'Closed'};
  const label = value => stageLabels[value]||String(value).replaceAll('_',' ');
  const tableStageLabels={needs_kb_update:'Needs Document Update',update_linked:'Documents Updated',ready_for_retest:'Ready to Test',closed:'Closed'};
  const tableStage = value => tableStageLabels[value]||label(value);
  const stageTone = value => ({needs_kb_update:'red',update_linked:'blue',ready_for_retest:'blue',closed:'green'}[value]||'blue');
  let testContext=null;
  const message = error => toast(error.message || 'Unable to complete request.', true);
  async function api(path, options = {}) {
    const response = await apiFetch('/admin' + path, options);
    const data = await response.json();
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Check the required fields and try again.');
    return data;
  }
  const docOptions = docs => '<option value="">Select document</option>' + docs.map(d => `<option value="${Number(d.id)}" title="${esc(documentDisplayTitle(d))}">#${Number(d.id)} ${esc(shortDocumentLabel(documentDisplayTitle(d)))}</option>`).join('');
  const field = (name, title, value = '') => `<label class="form-group">${esc(title)}<textarea id="${name}" maxlength="2000" required>${esc(value)}</textarea></label>`;
  const refresh = id => AdminPages.open(id);
  function correctionDocuments(row){
    try{const note=JSON.parse(row.retest_note);if(Array.isArray(note.documents))return note.documents;}catch(e){}
    return row.document_id?[{id:Number(row.document_id),title:row.document_title||'Linked document'}]:[];
  }
  function correctionDocumentLinks(row){
    const docs=correctionDocuments(row);
    return docs.length?docs.map(doc=>`<button class="correction-document-link" type="button" data-correction-document="${Number(doc.id)}">${U.icon('file')}<span>${documentTitleHtml(doc.title)}</span></button>`).join(''):'<span class="correction-not-linked">No linked sources</span>';
  }
  function bindDocs(host) {
    host.querySelectorAll('[data-maint-doc]').forEach(b => b.onclick = () => viewDoc(Number(b.dataset.maintDoc)));
  }

  const correctionsPage=AdminPages.register('corrections','Answer Corrections','Track a reported answer through KB changes and retesting. Original answers are preserved.', async (search,page) => {
    const rows = await api('/corrections');
    const stage=page.querySelector('#correction-stage-filter')?.value||'';
    const documentId=page.querySelector('#correction-document-filter')?.value||'';
    const range=page.querySelector('#correction-date-range')?.value||'';
    const now=Date.now(),withinRange=row=>!range||Date.parse(row.updated_at||'')>=now-Number(range)*86400000;
    const query=search.toLowerCase().trim();
    const documentOptions=[...new Map(rows.flatMap(correctionDocuments).map(doc=>[String(doc.id),doc.title])).entries()];
    const documentFilter=page.querySelector('#correction-document-filter');
    if(documentFilter){documentFilter.innerHTML=`<option value="">All documents</option><option value="__linked__">Linked documents</option>${documentOptions.map(([id,title])=>`<option value="${esc(id)}">${esc(title)}</option>`).join('')}`;documentFilter.value=documentId;}
    const filtered=rows.filter(row=>!stage||row.stage===stage).filter(row=>!documentId||(documentId==='__linked__'?correctionDocuments(row).length>0:correctionDocuments(row).some(doc=>String(doc.id)===documentId))).filter(withinRange).filter(row=>!query||[row.message_id,row.note,row.stage,row.document_title,row.answer_preview].join(' ').toLowerCase().includes(query));
    const size=Number(page.querySelector('#correction-page-size')?.value||adminPageSize()),totalPages=Math.max(1,Math.ceil(filtered.length/size));
    const current=Math.min(Math.max(Number(page.dataset.offset||0),0),totalPages-1),start=current*size,visible=filtered.slice(start,start+size);
    const count=stageName=>rows.filter(row=>row.stage===stageName).length;
    const linked=rows.filter(row=>correctionDocuments(row).length).length;
    const dateTime=value=>{const date=new Date(value);return Number.isNaN(date.getTime())?String(value||'Unavailable'):`${date.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'})}<small>${date.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'})}</small>`;};
    const cards=[['flag','Open Corrections',rows.filter(row=>row.stage!=='closed').length,'Waiting for document updates','open'],['chart','Ready to Test',count('ready_for_retest'),'Documents updated; waiting for a test','ready_for_retest'],['check','Closed',count('closed'),'Successfully resolved','closed'],['file','Linked Documents',linked,'Knowledge base documents linked','linked']];
    const tableRow=row=>{const linkedDocument=correctionDocumentLinks(row);return `<tr><td data-label="Answer / Question"><strong>Answer #${Number(row.message_id)}</strong><span>${esc(row.answer_preview||'Original answer preview not recorded')}</span></td><td data-label="Stage">${U.badge(tableStage(row.stage),stageTone(row.stage))}</td><td data-label="Linked Document">${linkedDocument}</td><td data-label="Issue">${esc(row.note||'No issue description recorded')}</td><td data-label="Updated">${dateTime(row.updated_at)}</td><td data-label="Action"><button class="btn btn-sm" type="button" data-correction="${Number(row.message_id)}">Open</button></td></tr>`;};
    const emptyTitle=rows.length?'No corrections match your current filters.':'No corrections yet.';
    const emptyText=rows.length?'Try resetting the filters to see all correction records.':'Start a correction from an answer in Review Queue when an answer requires a knowledge-base update.';
    const emptyAction=rows.length?'Reset Filters':'Go to Review Queue';
    const rowsHtml=visible.length?visible.map(tableRow).join(''):`<tr><td colspan="6"><div class="correction-empty"><strong>${emptyTitle}</strong><span>${emptyText}</span><button class="btn" type="button" id="correction-empty-action">${emptyAction}</button></div></td></tr>`;
    const pageStart=Math.max(0,Math.min(current-2,totalPages-5)),pageEnd=Math.min(totalPages,pageStart+5);
    const pageButtons=Array.from({length:pageEnd-pageStart},(_,index)=>{const number=pageStart+index;return `<button type="button" class="${number===current?'active':''}" data-correction-page="${number}" aria-current="${number===current?'page':'false'}>${number+1}</button>`;}).join('');
    return {html:`<section class="correction-summary-cards" aria-label="Correction summary">${cards.map(([icon,title,value,note,filter])=>`<button type="button" class="correction-summary-card" data-correction-filter="${filter}"><span>${U.icon(icon)}</span><div><strong>${Number(value).toLocaleString()}</strong><b>${esc(title)}</b><small>${esc(note)}</small></div><i aria-hidden="true">›</i></button>`).join('')}</section><section class="correction-table-card"><header><div><h2>Corrections</h2><p>Showing the most recently updated correction records.</p></div><strong>${filtered.length} ${filtered.length===1?'result':'results'}</strong></header><div class="tbl-wrap"><table class="correction-records-table"><thead><tr><th>Answer / Question</th><th>Stage</th><th>Linked Document</th><th>Issue</th><th>Updated</th><th>Action</th></tr></thead><tbody>${rowsHtml}</tbody></table></div><footer class="correction-pager"><span>Showing ${filtered.length?start+1:0}&ndash;${Math.min(start+size,filtered.length)} of ${filtered.length} corrections</span><nav aria-label="Corrections pagination"><button type="button" data-correction-page="${current-1}" ${current===0?'disabled':''}>‹ Previous</button>${pageButtons}<button type="button" data-correction-page="${current+1}" ${current>=totalPages-1?'disabled':''}>Next ›</button></nav><label>Rows per page:<select id="correction-page-size">${[7,10,20,50].map(pageSize=>`<option value="${pageSize}" ${size===pageSize?'selected':''}>${pageSize}</option>`).join('')}</select></label></footer></section>`,mount(host){
      host.querySelector('.correction-summary-cards').after(correctionFilters);
      host.querySelectorAll('[data-correction]').forEach(button=>button.onclick=()=>correction(Number(button.dataset.correction)));
      host.querySelectorAll('[data-correction-document]').forEach(button=>button.onclick=()=>AdminShell.go('knowledge').then(()=>AdminKnowledge.open(Number(button.dataset.correctionDocument))));
      host.querySelectorAll('[data-correction-page]').forEach(button=>button.onclick=()=>{if(button.disabled)return;page.dataset.offset=button.dataset.correctionPage;AdminPages.open('corrections');});
      host.querySelector('#correction-page-size').onchange=event=>{localStorage.setItem('laborlens_admin_page_size',event.target.value);page.dataset.offset='0';AdminPages.open('corrections');};
      host.querySelectorAll('[data-correction-filter]').forEach(button=>button.onclick=()=>{const value=button.dataset.correctionFilter;const stageFilter=page.querySelector('#correction-stage-filter');stageFilter.value=['ready_for_retest','closed'].includes(value)?value:'';page.querySelector('#correction-document-filter').value='';if(value==='linked')page.querySelector('#correction-document-filter').value='__linked__';page.dataset.offset='0';AdminPages.open('corrections');});
      const empty=host.querySelector('#correction-empty-action');if(empty)empty.onclick=()=>{if(rows.length){page.querySelector('form').reset();}else AdminShell.go('reviews');};
    }};
  });
  correctionsPage.classList.add('corrections-reference-page');
  const correctionFilters=correctionsPage.querySelector('form');
  correctionFilters.className='workbench-filters correction-filter-panel';
  correctionFilters.innerHTML=`<label class="correction-search-field">Search<input type="search" maxlength="200" placeholder="Search questions, answers, or issues..."/></label><label>Stage<select id="correction-stage-filter"><option value="">All stages</option>${stages.map(value=>`<option value="${value}">${esc(tableStage(value))}</option>`).join('')}</select></label><label>Linked Document<select id="correction-document-filter"><option value="">All documents</option></select></label><label>Date Range<select id="correction-date-range"><option value="">All time</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option></select></label><button class="btn btn-primary" type="submit">${U.icon('search')} Search / Refresh</button><button class="btn" type="reset">Reset</button>`;

  async function correction(id) {
    try {
      const [saved,answer] = await Promise.all([api(`/corrections/${id}`),api(`/answer-reviews/${id}`)]);
      const row = saved || {stage:'needs_kb_update'};
      if(testContext?.id===id&&testContext.draft?.revision===(row.revision||0))Object.assign(row,testContext.draft);
      let retest={question:'',result:row.retest_note||'',evidence:''};
      try{const parsed=JSON.parse(row.retest_note);if(parsed&&typeof parsed==='object'&&'result' in parsed)retest=parsed;}catch(e){}

      U.drawer(`Correction for answer #${id}`,
        `<h3>Original Question</h3><p>${esc(answer.question||'Not recorded')}</p><p>Review note: ${esc(answer.admin_review_note||'Not recorded')}</p><div class="correction-stages">${stages.map(stage=>`<span class="${stage===row.stage?'active':''}">${label(stage)}</span>`).join('<span aria-hidden="true">→</span>')}</div><p>Test the question, then review the saved answer and its automatic source references before closing.</p>` +
        `<label class="form-group">Stage<select id="correction-stage">${stages.map(s => `<option value="${s}">${esc(label(s))}</option>`).join('')}</select></label>` +
        '<section><h3>Reference documents</h3><p>Run Test Chatbot. Documents returned as sources will be saved here automatically.</p>'+correctionDocumentLinks(row)+'</section>' +
        field('correction-note','Problem / Required Correction',row.note) + field('correction-question','Test Question',retest.question||answer.question) + field('correction-result','Latest Test Answer',retest.result) + field('correction-evidence','Test Sources and Notes',retest.evidence),
        '<button class="btn" id="correction-answer">View original answer</button><button class="btn" id="correction-test">Test Chatbot</button><button class="btn btn-primary" id="correction-save">Save correction</button>');
      document.getElementById('correction-stage').value = row.stage;
      ['correction-result','correction-evidence'].forEach(id=>document.getElementById(id).maxLength=12000);
      document.getElementById('correction-question').maxLength=4000;
      document.querySelectorAll('#doc-modal [data-correction-document]').forEach(button=>button.onclick=()=>{closeDocModal();AdminShell.go('knowledge').then(()=>AdminKnowledge.open(Number(button.dataset.correctionDocument)));});
      document.getElementById('correction-answer').onclick = () => AdminReviews.open(id);
      const draft=()=>({revision:row.revision||0,stage:document.getElementById('correction-stage').value,document_id:row.document_id||null,note:document.getElementById('correction-note').value,retest_note:JSON.stringify({...retest,question:document.getElementById('correction-question').value,result:document.getElementById('correction-result').value,evidence:document.getElementById('correction-evidence').value})});
      document.getElementById('correction-test').onclick = async event => {
        if(running)return toast('Wait for the current test to finish.',true);
        if(document.getElementById('correction-stage').value==='closed')return toast('Reopen this correction before running a new retest.',true);
        const button=event.currentTarget;button.disabled=true;
        try{
          const payload=draft();
          const saved=await api(`/corrections/${id}`,{method:'PUT',body:JSON.stringify(payload)});
          payload.revision=saved.revision??payload.revision+1;
          testContext={id,question:document.getElementById('correction-question').value||answer.question,draft:payload};
          history=[];playgroundResult=null;closeDocModal();AdminShell.go('playground');
        }catch(error){message(error);}finally{button.disabled=false;}
      };
      document.getElementById('correction-save').onclick = async event => {
        const button = event.currentTarget; button.disabled = true;
        try {
          const payload=draft(),retest=JSON.parse(payload.retest_note);
          if(payload.retest_note.length>12000)throw Error('Keep the combined test question, answer and sources under 12,000 characters.');
          if(payload.stage==='closed'&&(!retest.question.trim()||!retest.result.trim()||!retest.evidence.trim()))throw Error('Test question, answer and sources are required to close.');
          const value = await api(`/corrections/${id}`, {method:'PUT',body:JSON.stringify(payload)});
          if(testContext?.id===id)testContext=null;
          toast(value.message); closeDocModal(); refresh('corrections');
        } catch(error) { message(error); } finally { button.disabled = false; }
      };
    } catch(error) { message(error); }
  }


  let history = [], playgroundResult = null, running = false, playgroundEpoch = 0;
  const playgroundPage = AdminPages.register('playground','Test Chatbot','Test the current guest-chat pipeline without saving conversations to normal chat history.', async () => ({
    html: '<p class="playground-note">Tests use the live chatbot and count toward AI service limits. This chat is temporary and does not use a signed-in user’s chat history. When opened from a correction, the test answer and sources are saved to that correction.</p>' +
      '<section class="playground-test-card"><label class="form-group">Language<select id="playground-language"><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label><label class="form-group">Test question<textarea id="playground-question" maxlength="4000" rows="4" placeholder="Enter a question to test the current guest-chat pipeline..."></textarea></label><div class="playground-actions"><button class="btn btn-primary" id="playground-run">➤ Ask LaborLens</button>' + (testContext?'<button class="btn" id="playground-return">↩ Return to Correction</button>':'') + '<button class="btn" id="playground-clear">⌫ Clear conversation</button></div></section><section id="playground-result" class="playground-results" aria-live="polite"></section>',
    mount(host) {
      const output=host.querySelector('#playground-result'),run=host.querySelector('#playground-run');
      let lastQuestion='',requestError='';
      if(testContext){host.querySelector('#playground-question').value=testContext.question||'';host.querySelector('#playground-return').onclick=()=>{if(running)return toast('Wait for the test to finish.',true);AdminShell.go('corrections');correction(testContext.id);};}
      const render = () => {
        const messages=history.map(message=>`<article class="playground-message ${message.role==='assistant'?'assistant':'user'}"><span class="playground-message-avatar">${message.role==='assistant'?'✦':'U'}</span><div><header><strong>${message.role==='assistant'?'assistant':'user'}</strong></header><p>${esc(message.content)}</p></div></article>`).join('');
        const result=playgroundResult?`<section class="playground-response"><h2>Response</h2><p class="playground-response-meta">${esc(playgroundResult.response.source_match || playgroundResult.response.confidence_level || 'Not recorded')} <span>·</span> ${Number(playgroundResult.elapsed_ms).toLocaleString()} ms</p><details class="admin-technical"><summary>More Details</summary><pre>${esc(JSON.stringify(playgroundResult.diagnostics,null,2))}</pre></details><h3>Sources</h3><div class="playground-sources">${(playgroundResult.response.sources||[]).map(source=>{const url=safeUrl(source.source_url||source.url);return `<article class="playground-source-card"><strong>${documentTitleHtml(source.document_title||source.source||'Source')}</strong>${source.article_section||source.article?`<small>${esc(source.article_section||source.article)}</small>`:''}<p>${esc(source.excerpt||source.text||source.content||'Excerpt not recorded')}</p>${url?`<a target="_blank" rel="noopener noreferrer" href="${esc(url)}">↗ Open Source</a>`:''}</article>`;}).join('')||'<div class="playground-no-sources">No supporting sources were returned for this test.</div>'}</div></section>`:'';
        const error=requestError?`<section class="playground-error"><strong>Test request failed.</strong><span>${esc(requestError)}</span><button class="btn btn-sm" type="button" id="playground-retry">Retry</button></section>`:'';
        output.innerHTML=messages+error+result;
        output.querySelector('#playground-retry')?.addEventListener('click',()=>{host.querySelector('#playground-question').value=lastQuestion;run.click();});
      };
      render();run.disabled=running;
      host.querySelector('#playground-clear').onclick=()=>{if(running)return;history=[];playgroundResult=null;requestError='';lastQuestion='';render();};
      run.onclick=async()=>{
        const input=host.querySelector('#playground-question'),question=input.value.trim();if(!question||running)return;
        const epoch=playgroundEpoch;running=true;run.disabled=true;requestError='';lastQuestion=question;output.innerHTML='<div class="playground-loading">Testing...</div>';
        try {
          const result=await api('/playground',{method:'POST',body:JSON.stringify({message:question,language:host.querySelector('#playground-language').value,history:history.slice(-10)})});
          if(epoch!==playgroundEpoch)return;
          playgroundResult=result;history.push({role:'user',content:question},{role:'assistant',content:result.response.answer,sources:result.response.sources||[],intent:result.response.intent||null});history=history.slice(-10);input.value='';
          if(testContext?.draft){
            const context=testContext;
            const evidence=[
              'Tested: '+new Date().toISOString(),
              'Language: '+host.querySelector('#playground-language').value,
              'Source support: '+(result.response.source_match||result.response.confidence_level||'Not recorded'),
              'Response time: '+Number(result.elapsed_ms)+' ms',
              ...(result.response.sources||[]).map((source,index)=>JSON.stringify({source:index+1,title:source.document_title||source.source||source.filename||'Source',provision:source.article_section||source.article||'',url:source.source_url||source.url||'',excerpt:String(source.excerpt||source.text||source.content||'').slice(0,400)}))
            ];
            if(!(result.response.sources||[]).length)evidence.push('No supporting sources returned. This test is not proof of correctness.');
            const documents=[...new Map((result.response.sources||[]).filter(source=>Number.isSafeInteger(Number(source.doc_db_id))&&Number(source.doc_db_id)>0).map(source=>[Number(source.doc_db_id),{id:Number(source.doc_db_id),title:source.document_title||source.source||'Source'}])).values()];
            context.draft.document_id=documents.length===1?documents[0].id:null;
            context.draft.retest_note=JSON.stringify({question,result:String(result.response.answer||''),evidence:evidence.join('\n'),documents});
            context.question=question;
            try{
              if(context.draft.retest_note.length>12000)throw Error('Test record exceeds the 12,000-character limit. Return to Correction to shorten and save it.');
              const saved=await api(`/corrections/${context.id}`,{method:'PUT',body:JSON.stringify(context.draft)});
              context.draft.revision=saved.revision??context.draft.revision+1;
              toast('Test answer and sources saved to correction #'+context.id+'. Review them before closing.');
            }catch(error){
              requestError='The chatbot answered, but its retest was NOT saved: '+error.message;
              message(Error(requestError));
            }
          }
          render();
        }catch(error){requestError=error.message||'Check the test input and try again.';render();message(error);}finally{running=false;run.disabled=false;if(epoch===playgroundEpoch&&!host.isConnected&&document.getElementById('tab-playground').classList.contains('tab-active'))refresh('playground');}
      };
    }
  }));
  playgroundPage.querySelector('form').remove();

  const performancePage = AdminPages.register('performance','System Performance','See how long answers take and check system status.',async()=>{
    const results=await Promise.allSettled([api('/evaluation-summary?days='+adminReportPeriod()),api('/system-health')]);
    const summary=results[0].status==='fulfilled'?results[0].value:null;
    const health=results[1].status==='fulfilled'?results[1].value:null;
    const ms=value=>value==null||!Number.isFinite(Number(value))?'Not recorded':`${(Number(value)/1000).toFixed(1)} s`;
    const friendlyTime=value=>{const date=new Date(value);return value&&Number.isFinite(date.getTime())?date.toLocaleString(undefined,{month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit'}):'Unavailable';};
    const p=summary?.performance;
    const components=Array.isArray(health?.components)?health.components:[],providers=Array.isArray(health?.providers)?health.providers:[];
    const observation=name=>components.find(component=>component.name===name)?.status||'Unavailable / no observation';
    const ai=providers.some(row=>row.status==='Observed successes')?'Recent answers succeeded (not a live check)':providers.some(row=>row.configured)?'No requests recorded':'Not set up';
    const statusTone=value=>/connected|ready|success/i.test(String(value))?'green':/unavailable|error|fail/i.test(String(value))?'red':'blue';
    const serviceRows=[['AI Provider',ai],['Database',observation('Database')],['Knowledge Base',observation('Knowledge Base')]];
    const service=`<section class="performance-service-card"><h2>Service Status</h2>${U.table(['Service','Observation'],serviceRows.map(([name,status])=>[esc(name),U.badge(status,statusTone(status))]),'performance-service-table')}</section>`;
    const timing=summary&&p?`<section class="performance-timing"><div class="performance-metrics">${[['Average Response Time',ms(p.average),'chart'],['Middle Response Time',ms(p.median),'chart']].map(([label,value,icon])=>`<article><span>${U.icon(icon)}</span><div><small>${label}</small><strong>${esc(value)}</strong></div></article>`).join('')}</div><p>Response times from the last ${adminReportPeriod()} days. Missing times do not mean the service is offline.</p></section>`:`<section class="performance-timing performance-unavailable"><p>Stored performance metrics unavailable.</p></section>`;
    const componentsTable=components.length?U.table(['Component','Observation'],components.map(component=>[esc(component.name),U.badge(component.status,statusTone(component.status))]),'performance-component-table'):'<p class="performance-empty">No component observations recorded.</p>';
    const providerRows=providers.map(row=>[esc(`${row.provider||'Provider not recorded'} / ${row.model||'Model not recorded'}`),U.badge(row.status||'No observations',statusTone(row.status)),esc(row.requests??'Not recorded'),esc(row.successes??'Not recorded'),esc(row.failures??'Not recorded'),esc(row.timeouts??'Not recorded'),esc(ms(row.average_latency_ms))]);
    const providersTable=providerRows.length?U.table(['Provider / model','Status','Attempts','Successes','Failures','Timeouts','Average response time'],providerRows,'performance-provider-table'):'<p class="performance-empty">No provider observations recorded since the current backend process started.</p>';
    const technical=`<details class="performance-technical"><summary>More Details</summary><div class="performance-technical-body">${summary&&p?`<p class="performance-note">Timing samples: ${num(p.samples).toLocaleString()}. ${esc(summary.note||'Recorded response timing statistics are based on query logs.')}</p>`:''}<section class="performance-observations"><h3>Service details</h3>${componentsTable}</section><section class="performance-providers"><h3>AI service details</h3><p>${esc(health?.note||'AI service details are unavailable.')} Includes admin tests; attempts are not unique conversations.</p>${providersTable}</section></div></details>`;
    return service+timing+`<p class="performance-health">Last Health Check: <time title="${esc(health?.checked_at||'Unavailable')}">${esc(friendlyTime(health?.checked_at))}</time></p>`+technical;
  });

  const performanceForm=performancePage.querySelector('form');
  performanceForm.classList.add('performance-refresh-bar');
  performanceForm.querySelector('label')?.remove();
  performanceForm.querySelector('button[type="reset"]')?.remove();
  const performanceButton=performanceForm.querySelector('button');
  performanceButton.classList.add('btn-primary');
  performanceButton.innerHTML='↻ Refresh performance';
  let performanceRefreshing=false;
  performanceForm.onsubmit=async event=>{event.preventDefault();if(performanceRefreshing)return;performanceRefreshing=true;performanceButton.disabled=true;performanceButton.textContent='Refreshing…';try{await AdminPages.open('performance');}finally{performanceRefreshing=false;performanceButton.disabled=false;performanceButton.innerHTML='↻ Refresh performance';}};

  const coveragePage=AdminPages.register('coverage','Knowledge Base Coverage','Review searchable text sections, document readiness, and category coverage.',async search=>{
    const [stats,documents]=await Promise.all([AdminPages.api('/kb/stats'),AdminPages.api('/kb/documents')]);
    const summary=kbSummary(stats||{}),docs=Array.isArray(documents)?documents:[];
    const ready=docs.filter(doc=>documentProcessingStatus(doc)==='Ready');
    const processing=docs.filter(doc=>['Pending','Processing','Reading text','Splitting text','Preparing for search','Replacing'].includes(documentProcessingStatus(doc)));
    const attention=docs.filter(doc=>documentProcessingStatus(doc)==='Failed'||(String(doc.status||'').toLowerCase()==='indexed'&&Number(doc.chunk_count||0)===0));
    const query=String(search||'').trim().toLowerCase();
    const visible=docs.filter(doc=>!query||[documentDisplayTitle(doc),doc.category,doc.primary_category,doc.issuing_agency,documentProcessingStatus(doc)].join(' ').toLowerCase().includes(query));
    const categoryCounts=new Map();
    docs.forEach(doc=>{const category=doc.primary_category||doc.category||'Uncategorized';categoryCounts.set(category,(categoryCounts.get(category)||0)+1);});
    const categoryRows=[...categoryCounts.entries()].sort((a,b)=>b[1]-a[1]);
    const statusRows=[['Document search',summary.chunks?`${summary.chunks.toLocaleString()} searchable sections available`:'No searchable sections available'],['Ready documents',`${ready.length.toLocaleString()} of ${docs.length.toLocaleString()}`],['Categories with documents',`${summary.activeCategories||categoryRows.length} active`],['Search model',stats?.chroma?.embedding_model||'Not recorded']];
    const rows=visible.map(doc=>`<tr><td data-label="Document">${documentTitleHtml(documentDisplayTitle(doc))}<small>${esc(doc.issuing_agency||'Issuing agency not recorded')}</small></td><td data-label="Category">${esc(doc.primary_category||doc.category||'Uncategorized')}</td><td data-label="Status">${U.badge(documentProcessingStatus(doc),documentProcessingStatus(doc)==='Ready'?'green':documentProcessingStatus(doc)==='Failed'?'red':'blue')}</td><td data-label="Searchable Sections">${Number(doc.chunk_count||0).toLocaleString()}</td><td data-label="Updated">${esc(dateLabel(doc.updated_at||doc.last_indexed_at||doc.created_at))}</td><td data-label="Action"><button class="btn btn-sm" type="button" data-coverage-document="${Number(doc.id)}">View</button></td></tr>`).join('');
    return {html:`<section class="coverage-summary-cards" aria-label="Knowledge Base coverage summary"><article><small>Total Documents</small><strong>${docs.length.toLocaleString()}</strong><span>All saved document records</span></article><article><small>Total Searchable Sections</small><strong>${summary.chunks.toLocaleString()}</strong><span>Text sections ready for retrieval</span></article><article><small>Documents Ready</small><strong>${ready.length.toLocaleString()}</strong><span>Available to support answers</span></article><article><small>Needs Attention</small><strong>${attention.length.toLocaleString()}</strong><span>${processing.length.toLocaleString()} currently processing</span></article></section><section class="coverage-details-grid"><section class="panel"><div class="panel-hdr"><h2 class="panel-title">Document Search Details</h2></div><div class="panel-body">${U.table(['Detail','Current value'],statusRows.map(([label,value])=>[esc(label),esc(value)]),'coverage-status-table')}</div></section><section class="panel"><div class="panel-hdr"><h2 class="panel-title">Documents by Category</h2></div><div class="panel-body">${categoryRows.length?U.bars(categoryRows.map(([label,count])=>({label,count}))):'<p class="empty">No document categories recorded.</p>'}</div></section></section><section class="coverage-documents-card"><header><div><h2>Document Coverage</h2><p>${visible.length===docs.length?`${docs.length.toLocaleString()} document records`:`${visible.length.toLocaleString()} matching document records`}</p></div><span>${summary.chunks.toLocaleString()} total searchable sections</span></header><div class="tbl-wrap"><table class="coverage-documents-table"><thead><tr><th>Document</th><th>Category</th><th>Status</th><th>Searchable Sections</th><th>Updated</th><th>Action</th></tr></thead><tbody>${rows||'<tr><td colspan="6" class="empty">No documents match this search.</td></tr>'}</tbody></table></div></section>`,mount(host){host.querySelectorAll('[data-coverage-document]').forEach(button=>button.onclick=()=>AdminShell.go('knowledge').then(()=>AdminKnowledge.open(Number(button.dataset.coverageDocument))));}};
  });
  coveragePage.classList.add('coverage-reference-page');
  coveragePage.querySelector('input[type="search"]').placeholder='Search document coverage...';

  AdminPages.register('faqs','Common Questions','Create and maintain structured FAQs in English, Filipino and Hiligaynon.',async search=>{
    const rows=await api('/faqs');
    return {html:'<button class="btn btn-primary" id="faq-create">Add FAQ</button>'+U.table(['Question','Language','Category','Updated','Action'],rows.filter(r=>[r.question,r.answer,r.category].join(' ').toLowerCase().includes(search.toLowerCase())).map(r=>[
      esc(r.question),esc(languageLabel(r.language)),esc(r.category||'Uncategorized'),esc(dateLabel(r.updated_at)),`<button class="btn btn-sm" data-faq="${Number(r.id)}">Edit</button>`
    ])),mount(host){host.querySelector('#faq-create').onclick=()=>faqEditor();host.querySelectorAll('[data-faq]').forEach(b=>b.onclick=()=>faqEditor(rows.find(r=>r.id===Number(b.dataset.faq))));}};
  });
  function faqEditor(row={}) {
    U.drawer(row.id?'Edit FAQ':'Add FAQ',field('faq-question','Question',row.question)+field('faq-answer','Approved answer',row.answer)+
      `<label class="form-group">Category<select id="faq-category"><option value="">Uncategorized</option>${[...new Set([...KB_CATEGORIES,...(row.category?[row.category]:[])])].map(c=>`<option>${esc(c)}</option>`).join('')}</select></label><label class="form-group">Language<select id="faq-language"><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label>`,
      (row.id?'<button class="btn btn-danger" id="faq-delete">Delete FAQ</button>':'')+'<button class="btn btn-primary" id="faq-save">Save FAQ</button>');
    document.getElementById('faq-category').value=row.category||'';document.getElementById('faq-language').value=row.language||'en';
    document.getElementById('faq-save').onclick=async event=>{
      const b=event.currentTarget,question=document.getElementById('faq-question').value.trim(),answer=document.getElementById('faq-answer').value.trim();if(!question||!answer){toast('Question and answer are required.',true);return;}b.disabled=true;
      try{await api('/faqs'+(row.id?`/${row.id}`:''),{method:row.id?'PUT':'POST',body:JSON.stringify({question,answer,category:document.getElementById('faq-category').value||null,language:document.getElementById('faq-language').value})});closeDocModal();refresh('faqs');toast('FAQ saved.');}catch(error){message(error);}finally{b.disabled=false;}
    };
    if(row.id)document.getElementById('faq-delete').onclick=async event=>{
      if(!confirm('Permanently delete this FAQ? This cannot be undone here.'))return;const b=event.currentTarget;b.disabled=true;
      try{await api(`/faqs/${row.id}`,{method:'DELETE'});closeDocModal();refresh('faqs');toast('FAQ deleted.');}catch(error){message(error);}finally{b.disabled=false;}
    };
  }
  const logout=doLogout;
  doLogout=function(){playgroundEpoch++;testContext=null;history=[];playgroundResult=null;playgroundPage.querySelector('.page-results').replaceChildren();return logout();};
  return {correction};
})();
