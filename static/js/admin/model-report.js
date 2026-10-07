/* Read-only rendering of the existing admin health response. */
(function(root){
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString();
  const seconds=value=>value==null||!Number.isFinite(Number(value))?'Not recorded':`${(Number(value)/1000).toFixed(2)} s`;
  function renderScheduler(data){
    if(!data)return '<section class="performance-scheduler performance-unavailable"><h3>Model Request Pool</h3><p>Scheduling details unavailable. Restart the updated backend, then refresh performance.</p></section>';
    if(data.enabled!==true)return '<section class="performance-scheduler"><h3>Model Request Pool</h3><p>Model pool scheduling is disabled. Requests use the configured provider order.</p></section>';
    if(data.concurrency_uncapped===true){
      const values=[['Primary models',number(data.primary_model_count)],['Backup models',number(data.backup_model_count)],['Active requests',number(data.active_requests)],['App concurrency cap','No local cap']];
      return `<section class="performance-scheduler" aria-label="Generation model capacity"><header><h3>Model Request Pool</h3><span class="performance-model-status">Snapshot · this backend worker</span></header><p>Primary models share requests: idle first, then least busy. Occupied models can handle additional requests immediately.</p><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl><p>Gemini backups are used only after the OpenAI primary models fail or are unavailable—not just because they are busy.</p><p>No app-local generation concurrency cap or capacity queue. Provider limits and existing request deadlines still apply. Refresh performance to update the counts.</p></section>`;
    }
    const values=[['Models in pool',number(data.model_count)],['Active requests',number(data.active_requests)],['Configured slots',number(data.configured_slots)],['Waiting for a slot',number(data.waiting_requests)]];
    return `<section class="performance-scheduler" aria-label="Generation model capacity"><header><h3>Model Request Pool</h3><span class="performance-model-status">Snapshot · this backend worker</span></header><p>Use an idle model first. If all are occupied, use the least-busy model. Configured priority breaks ties.</p><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl><p>Up to <strong>${number(data.concurrency_per_model)}</strong> active requests per model · wait up to <strong>${number(data.queue_wait_seconds)} seconds</strong> for a slot · maximum <strong>${number(data.queue_max_waiters)}</strong> waiting requests.</p><p>Refresh performance to update these counts. Configured slots are maximum capacity, not a guarantee that every model is usable. Provider limits and existing request deadlines still apply.</p></section>`;
  }
  function renderLda(data){
    const status=data?.status||'Unavailable';
    const ready=data?.ready===true&&['Ready','Matching chunks','Matching incomplete'].includes(status);
    const tone=status==='Ready'?'good':['Failed','Needs documents','Unavailable','Matching incomplete'].includes(status)?'warning':'neutral';
    const values=[['Topics learned',ready?number(data.topic_count):'—'],['Training chunks',ready?number(data.trained_chunk_count):'—'],['Topic selection',data?.topic_selection||'—'],['Search hints',data?(data.search_hints_enabled?(ready?'On':'Waiting'):'Off'):'—']];
    const coverage=data?.assignment_checked_chunks!=null?`<p class="performance-model-detail">Chunks checked: <strong>${number(data.assignment_checked_chunks)} / ${number(data.assignment_total_chunks)}</strong> · With topic matches: ${number(data.assignment_classified_chunks)} · No clear match: ${number(data.assignment_unclassified_chunks)}</p>`:'';
    return `<section class="performance-model-group"><header><h3>Topic Learning (LDA)</h3><p>Helps find related document passages. It does not write or check answers.</p></header><article class="performance-model-card performance-lda-card"><div class="performance-model-heading"><span>KNOWLEDGE BASE</span><span class="performance-model-status ${tone}" aria-live="polite">${escape(status)}</span></div><h4>Document topics</h4><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>${coverage}<p class="performance-model-detail">${escape(data?.detail||'Topic-learning status is unavailable. Refresh the page after restarting the backend.')}</p><p class="performance-model-detail">Topics are learned from a document sample; all chunks are then checked. A chunk can match several topics, with no top-10 cutoff. Unclear matches never block search. Status is for this backend worker and resets after restart.</p></article></section>`;
  }
  function generationStatus(row){
    if(row.enabled===false)return 'Disabled';
    if(row.configured===false)return 'Not configured';
    return row.status||'No status reported';
  }
  function generationSummary(health){
    if(!health)return 'Status unavailable';
    const rows=(Array.isArray(health.providers)?health.providers:[]).filter(row=>(row.role||'generation')==='generation');
    if(!rows.length)return 'No generation models reported';
    const eligible=rows.filter(row=>row.enabled!==false&&row.configured!==false);
    if(!eligible.length)return 'No enabled, configured models';
    if(eligible.some(row=>generationStatus(row)==='In use'))return 'Generating a response';
    if(eligible.some(row=>/fail|timed out|limit reached|waiting|incomplete|interrupted/i.test(generationStatus(row))))return 'Model issues reported — see details';
    if(eligible.some(row=>generationStatus(row)==='Observed successes'))return 'Successful model calls recorded';
    return 'No successful model calls reported';
  }
  function render(health){
    if(!health)return '<section class="performance-model-group"><h2>Answer Generation</h2><p>Model information is unavailable. Try Refresh.</p></section>';
    const models=(Array.isArray(health.providers)?health.providers:[]).filter(row=>(row.role||'generation')==='generation')
      .sort((a,b)=>(Number(a.priority)>0?Number(a.priority):Infinity)-(Number(b.priority)>0?Number(b.priority):Infinity));
    const items=models.map(row=>{
      const status=generationStatus(row);
      const tone=/fail|missing|not configured|timed out|limit reached|waiting|incomplete|interrupted/i.test(status)?'warning':status==='Observed successes'?'good':'neutral';
      const observed=row.metrics_available!==false;
      const metric=key=>observed?number(row[key]):'—';
      const completed=Number(row.successes)+Number(row.failures);
      const latency=observed&&Number.isFinite(completed)&&completed>0?seconds(row.average_latency_ms):'Not recorded';
      const values=[['Model calls',metric('requests')],['Successful calls',metric('successes')],['Failed calls',metric('failures')],['Average completed call',latency]];
      const details=[];
      if(Number(row.priority)>0)details.push('Configured preference '+number(row.priority));
      if(observed&&row.timeouts!=null)details.push('Timeouts '+number(row.timeouts));
      if(observed&&row.rate_limits!=null)details.push('Rate-limit errors '+number(row.rate_limits));
      if(row.retry_in_seconds>0)details.push('Retry in about '+number(row.retry_in_seconds)+' seconds');
      if(!observed)details.push('Metrics unavailable');
      return `<li class="performance-model-row"><div class="performance-model-entry"><div class="performance-model-line"><span class="performance-model-provider">${escape(String(row.provider||'').toUpperCase())}</span><h4>${escape(row.model||'Model not reported')}</h4><span class="performance-model-status ${tone}">${escape(status)}</span></div><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl><p class="performance-model-detail">${details.map(escape).join(' · ')}</p></div></li>`;
    }).join('');
    return `<section class="performance-models">${renderScheduler(health.generation_scheduler)}<section class="performance-model-group"><header><h2>Answer Generation</h2><p>Models used to draft responses. The backend reports these observations since its last restart.</p><p class="performance-note">Successful model calls are not a count of final answers delivered or a measure of answer accuracy. One question can involve several model calls.</p><p class="performance-model-order-note">Listed by configured preference; availability, limits and retries can change which model handles a request.</p></header>${models.length?`<ul class="performance-model-list" aria-label="Answer generation models">${items}</ul>`:'<p class="performance-model-empty">No generation models reported by the backend.</p>'}</section>${renderLda(health.lda)}</section>`;
  }
  root.AdminModelReport={render,generationSummary};
})(globalThis);
