/* Settings overview with safe server configuration and browser-only preferences. */
function settingsCard(id,title,description,icon){return `<button class="settings-card" type="button" data-setting="${id}" aria-label="Open ${esc(title)} settings"><span class="settings-card-icon">${AdminUI.icon(icon)}</span><span class="settings-card-copy"><strong>${esc(title)}</strong><small>${esc(description)}</small></span><b aria-hidden="true">›</b></button>`;}
function settingsValue(value){return value===undefined||value===null||value===''?'Not available':String(value);}
function settingsList(rows){return `<dl class="settings-list">${rows.map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${esc(settingsValue(value))}</dd></div>`).join('')}</dl>`;}
function settingsComponent(components,name){return components.find(component=>component.name===name);}
function settingsConfigField(field,values){
 const value=values[field.key],help=field.help?`<small>${esc(field.help)}</small>`:'';
 if(field.kind==='pool')return `<label class="settings-config-field settings-config-wide">${esc(field.label)}<textarea data-config-key="${field.key}" rows="4" spellcheck="false">${esc(String(value||'').split(',').join('\n'))}</textarea>${help}</label>`;
 if(field.kind==='boolean')return `<label class="settings-config-toggle"><input type="checkbox" data-config-key="${field.key}" ${value?'checked':''}><span>${esc(field.label)}</span>${help}</label>`;
 if(field.kind==='text')return `<label class="settings-config-field">${esc(field.label)}<input type="text" data-config-key="${field.key}" maxlength="80" value="${esc(settingsValue(value))}">${help}</label>`;
 const step=field.kind==='number'?'any':'1';
 return `<label class="settings-config-field">${esc(field.label)}<input type="number" data-config-kind="${field.kind}" data-config-key="${field.key}" min="${field.min}" max="${field.max}" step="${step}" value="${esc(settingsValue(value))}">${help}</label>`;
}
function settingsConfigForm(id,values,sections){return `<form class="settings-detail-block settings-config-form" data-config-form="${id}">${sections.map(section=>`<section class="settings-config-section"><div class="settings-detail-heading"><h2>${esc(section.title)}</h2><span class="settings-editable">Editable</span></div>${section.description?`<p class="settings-detail-note">${esc(section.description)}</p>`:''}<div class="settings-config-grid">${section.fields.map(field=>settingsConfigField(field,values)).join('')}</div></section>`).join('')}<p class="settings-detail-note">Provider keys, passwords, and live system status cannot be changed here. Saved values are stored on this server, apply to this worker immediately, and load after a backend restart. Model IDs must be supported by the provider and have a configured key.</p><p class="settings-save-message" data-save-status role="status" aria-live="polite"></p><button class="btn btn-primary" type="submit">Save Settings</button><button class="btn" type="button" data-reset-settings>Restore Deployment Defaults</button></form>`;}

const settingsPage=AdminPages.register('settings','Settings','Manage LaborLens configuration and admin preferences.',async()=>{
 const [config,health,account,documentsData,categoriesData]=await Promise.all([
  AdminPages.api('/admin/configuration'),
  AdminPages.api('/admin/system-health').catch(()=>({components:[],providers:[]})),
  AdminPages.api('/auth/me').catch(()=>currentUser||{}),
  AdminPages.api('/kb/documents').catch(()=>null),
  AdminPages.api('/kb/categories').catch(()=>null)
 ]);
  let values=config?.values||{};const components=Array.isArray(health?.components)?health.components:[],providers=Array.isArray(health?.providers)?health.providers:[];
 const documents=Array.isArray(documentsData)?documentsData:null,categories=Array.isArray(categoriesData?.categories)?categoriesData.categories:null;
 const chroma=settingsComponent(components,'Chroma'),knowledge=settingsComponent(components,'Knowledge Base'),embedding=settingsComponent(components,'Embedding service');
 const lastUpdate=documents?.map(document=>document.last_indexed_at||document.updated_at||document.created_at).filter(Boolean).sort((a,b)=>new Date(b)-new Date(a))[0];
 const dateValue=lastUpdate?new Date(lastUpdate).toLocaleString('en-US',{dateStyle:'medium',timeStyle:'short'}):'Not available';
  const readOnly='<span class="settings-readonly">Read only · Live system information</span>';
 const home=()=>`<section class="settings-home"><div class="settings-grid">${[
  ['general','General','Application and system information.','settings'],
  ['ai','AI Models','Model pools, provider availability, and answer-generation settings.','chat'],
  ['retrieval','Retrieval','Search breadth, evidence size, and search-cache settings.','search'],
  ['knowledge','Knowledge Base','Processing, storage, and document information.','folder'],
  ['preferences','Admin Preferences','Account and display preferences for this browser.','users']
  ].map(card=>settingsCard(...card)).join('')}</div><aside class="settings-note"><span>${AdminUI.icon('file')}</span><div><strong>Safe settings</strong><p>AI models, retrieval limits, and response budgets can be changed here. Provider keys, passwords, and live system status remain protected.</p></div></aside></section>`;
 const details={
   general:{description:'Edit the application name and review basic system information.',body:()=>`${settingsConfigForm('general',values,[{title:'Application',fields:[{key:'APP_NAME',label:'Application Name',kind:'text',help:'Shown in the API documentation and admin configuration.'}]}])}<section class="settings-detail-block"><div class="settings-detail-heading"><h2>Security & System Information</h2>${readOnly}</div>${settingsList([['Application Description','AI-powered Philippine Labor Law Assistant'],['System Version','Not available'],['Supported Languages','English, Filipino, Hiligaynon'],['Admin Authentication',account?.role==='admin'?'Enabled':'Unavailable'],['Session Expiration',values.JWT_EXPIRE_MINUTES?`${values.JWT_EXPIRE_MINUTES} minutes`:'Not available'],['Secret Values','Hidden'],['Signed-in Role',account?.role||'Not available']])}</section>`},
   ai:{description:'Edit the configured model pools, provider availability, and response limits.',body:()=>settingsConfigForm('ai',values,[
    {title:'Model Pools',description:'One provider/model per line, in attempt order. No five-model cap; up to 32 supported model IDs per pool.',fields:[
     {key:'LABORLENS_GENERATOR_POOL',label:'Answer-generation models',kind:'pool'},
     {key:'LABORLENS_RERANKER_POOL',label:'Passage-ranking models',kind:'pool'},
     {key:'LABORLENS_VERIFIER_POOL',label:'Answer-checking models',kind:'pool'},
     {key:'LABORLENS_QUERY_POOL',label:'Query-understanding models',kind:'pool'},
     {key:'LABORLENS_MODEL_POOLS_ENABLED',label:'Use configured model pools',kind:'boolean',help:'Turn this off to use the legacy primary/fallback provider route.'},
     {key:'LABORLENS_OPENAI_ENABLED',label:'Allow OpenAI models',kind:'boolean'},
     {key:'LABORLENS_GEMINI_ENABLED',label:'Allow Gemini models',kind:'boolean'}]},
    {title:'Answer and Timing Limits',fields:[
     {key:'LABORLENS_MAX_OUTPUT_TOKENS',label:'Normal answer token limit',kind:'integer',min:400,max:2048},
     {key:'LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX',label:'Detailed answer token limit',kind:'integer',min:400,max:2048},
     {key:'LABORLENS_SIMPLE_OUTPUT_TOKENS',label:'Short answer token limit',kind:'integer',min:400,max:2048},
     {key:'LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS',label:'Normal request budget (seconds)',kind:'number',min:10,max:120},
     {key:'LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS',label:'Complex request budget (seconds)',kind:'number',min:10,max:180},
     {key:'LABORLENS_OPENAI_TIMEOUT_SECONDS',label:'OpenAI attempt timeout (seconds)',kind:'number',min:5,max:30},
     {key:'LABORLENS_GEMINI_TIMEOUT_SECONDS',label:'Gemini attempt timeout (seconds)',kind:'number',min:5,max:30},
     {key:'LABORLENS_GROQ_TIMEOUT_SECONDS',label:'Groq attempt timeout (seconds)',kind:'number',min:5,max:30}]}])},
   retrieval:{description:'Edit how broadly LaborLens searches and how much evidence it sends to answer generation.',body:()=>settingsConfigForm('retrieval',values,[
    {title:'Search and Evidence',fields:[
     {key:'RAG_INITIAL_TOP_K',label:'Initial search passages',kind:'integer',min:4,max:30},
     {key:'RAG_EXPANDED_TOP_K',label:'Wider search passages',kind:'integer',min:4,max:50,help:'Must be at least the initial search count.'},
     {key:'RAG_CANDIDATE_POOL_SIZE',label:'Candidate passages for ranking',kind:'integer',min:10,max:50},
     {key:'RAG_CONTEXT_TOKEN_BUDGET',label:'Evidence context token budget',kind:'integer',min:256,max:30000},
     {key:'LABORLENS_LLM_RERANK_ENABLED',label:'Use model-based passage reranking',kind:'boolean'},
     {key:'LABORLENS_CONDITIONAL_RERANK_ENABLED',label:'Skip reranking when evidence is already clear',kind:'boolean'},
     {key:'LABORLENS_QUERY_REFORMULATION_ENABLED',label:'Use query reformulation for difficult searches',kind:'boolean'},
     {key:'LDA_RERANK_ENABLED',label:'Use learned topic hints in ranking',kind:'boolean'},
     {key:'LDA_SEARCH_HINTS_ENABLED',label:'Use learned topic hints in search',kind:'boolean'}]},
    {title:'Search Cache',description:'A larger cache uses more memory; a longer duration may reuse older search results.',fields:[
     {key:'LABORLENS_RETRIEVAL_CACHE_SIZE',label:'Maximum cached searches',kind:'integer',min:16,max:2048},
     {key:'LABORLENS_RETRIEVAL_CACHE_TTL_SECONDS',label:'Cache duration (seconds)',kind:'number',min:0,max:86400}]}])},
  knowledge:{description:'View Knowledge Base processing and retrieval information.',body:`<section class="settings-detail-block"><div class="settings-detail-heading"><h2>Knowledge Base Status</h2>${readOnly}</div>${settingsList([['Total Documents',documents?documents.length.toLocaleString():'Not available'],['Total Categories',categories?categories.length.toLocaleString():'Not available'],['Retrieval',knowledge?.status],['Document Processing',documentsData===null?'Unavailable':'Available'],['Supported Files','PDF, DOCX, PNG, JPG, TIFF, BMP, WebP'],['Vector Store',chroma?'Chroma':'Not available'],['Vector Store Status',chroma?.status],['Embedding Service',embedding?.status],['Stored Text Sections',knowledge?.chunks],['Last Knowledge Base Update',dateValue]])}<p class="settings-detail-note">Document uploads, replacements, categories, and metadata remain in Knowledge Base. No reset or rebuild actions are available here.</p></section>`}
 };
 return {html:home(),mount(host){
  const renderHome=()=>{host.innerHTML=home();host.querySelectorAll('[data-setting]').forEach(button=>button.onclick=()=>renderDetail(button.dataset.setting));};
  const renderPreferences=()=>{
   const density=localStorage.getItem('laborlens_admin_density')||'comfortable',pageSize=localStorage.getItem('laborlens_admin_page_size')||'10',reportPeriod=localStorage.getItem('laborlens_default_report_period')||'30';
   host.innerHTML=`<section class="settings-detail"><button class="settings-back" type="button">← Back to Settings</button><p class="settings-detail-description">Adjust supported interface and admin display preferences.</p><form id="admin-preferences-form" class="settings-detail-block settings-preferences"><div class="settings-detail-heading"><h2>Display Preferences</h2><span class="settings-editable">Editable in this browser</span></div><fieldset><legend>Table Density</legend><label><input type="radio" name="density" value="comfortable" ${density==='comfortable'?'checked':''}> Comfortable</label><label><input type="radio" name="density" value="compact" ${density==='compact'?'checked':''}> Compact</label></fieldset><label>Default Rows Per Page<select name="page_size"><option value="7">7</option><option value="10">10</option><option value="20">20</option><option value="50">50</option></select></label><label>Default Report Period<select name="report_period"><option value="7">Last 7 Days</option><option value="30">Last 30 Days</option></select></label>${settingsList([['Confirmation Dialogs','Enabled']])}<p class="settings-detail-note">Confirmation dialogs stay enabled to protect sensitive actions.</p><button class="btn btn-primary" type="submit">Save Preferences</button></form><section class="settings-detail-block"><div class="settings-detail-heading"><h2>My Account</h2><span class="settings-editable">Editable</span></div><p class="settings-detail-note">Change your LaborLens sign-in details. This does not change your Google/Gmail password.</p><form id="account-settings-form" class="settings-account-form"><label>Email Address<input id="account-email" type="email" autocomplete="email" maxlength="255" required value="${esc(account.email||'')}"></label><label>Current LaborLens Password<input id="account-current-password" type="password" autocomplete="current-password" required></label><label>New Password (optional)<input id="account-new-password" type="password" autocomplete="new-password" minlength="8" maxlength="72"></label><label>Confirm New Password<input id="account-confirm-password" type="password" autocomplete="new-password" maxlength="72"></label><small>Use at least 8 characters. Leave the new-password fields blank to keep your current password.</small><p id="account-save-message" role="status" aria-live="polite"></p><button class="btn" type="submit">Save Account</button></form></section></section>`;
   host.querySelector('.settings-back').onclick=renderHome;
   const preferences=host.querySelector('#admin-preferences-form');preferences.elements.page_size.value=pageSize;preferences.elements.report_period.value=reportPeriod;
   preferences.onsubmit=event=>{event.preventDefault();const data=new FormData(preferences),newDensity=String(data.get('density')),newPageSize=String(data.get('page_size')),newPeriod=String(data.get('report_period'));localStorage.setItem('laborlens_admin_density',newDensity);localStorage.setItem('laborlens_admin_page_size',newPageSize);localStorage.setItem('laborlens_default_report_period',newPeriod);document.getElementById('dashboard').dataset.density=newDensity;toast('Admin preferences saved.');};
   const accountForm=host.querySelector('#account-settings-form'),message=host.querySelector('#account-save-message');
   accountForm.onsubmit=async event=>{event.preventDefault();const email=accountForm.querySelector('#account-email').value.trim(),password=accountForm.querySelector('#account-new-password').value,confirm=accountForm.querySelector('#account-confirm-password').value;if(password!==confirm){message.textContent='New passwords do not match.';return;}if(password&&new TextEncoder().encode(password).length>72){message.textContent='New password is too long.';return;}const button=accountForm.querySelector('button');button.disabled=true;message.textContent='Saving…';try{const response=await apiFetch('/auth/me/account',{method:'PATCH',body:JSON.stringify({email,current_password:accountForm.querySelector('#account-current-password').value,...(password?{new_password:password}:{})})});const result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'Unable to save your account.');if(result.reauthenticate){doLogout();toast('Password updated. Please sign in again.');return;}account.email=result.email;if(currentUser){currentUser.email=result.email;saveAdminSession();}accountForm.querySelector('#account-email').value=result.email;message.textContent=result.message;toast('Account settings saved.');}catch(error){message.textContent=error.message||'Unable to save your account.';}finally{accountForm.querySelectorAll('input[type="password"]').forEach(input=>input.value='');button.disabled=false;}};
  };
  const renderDetail=id=>{if(id==='preferences')return renderPreferences();const detail=details[id];if(!detail)return renderHome();host.innerHTML=`<section class="settings-detail"><button class="settings-back" type="button">← Back to Settings</button><p class="settings-detail-description">${esc(detail.description)}</p>${typeof detail.body==='function'?detail.body():detail.body}</section>`;host.querySelector('.settings-back').onclick=renderHome;wireConfigurationForms(id);};
  const wireConfigurationForms=id=>{
   const form=host.querySelector('[data-config-form]');if(!form)return;
   const message=form.querySelector('[data-save-status]'),save=form.querySelector('button[type="submit"]');
   form.onsubmit=async event=>{
    event.preventDefault();const changes={};
    form.querySelectorAll('[data-config-key]').forEach(field=>{
     const key=field.dataset.configKey;let value;
     if(field.type==='checkbox')value=field.checked;
     else if(field.dataset.configKind==='integer')value=Number.parseInt(field.value,10);
     else if(field.dataset.configKind==='number')value=Number(field.value);
     else if(field.tagName==='TEXTAREA')value=field.value.split(/[\n,]+/).map(item=>item.trim()).filter(Boolean);
     else value=field.value;
     const current=values[key];let unchanged;
     if(Array.isArray(value)){
      const existing=Array.isArray(current)?current:String(current||'').split(/[\n,]+/).map(item=>item.trim()).filter(Boolean);
      unchanged=existing.join(',')===value.join(',');
     }else if(typeof value==='boolean')unchanged=Boolean(current)===value;
     else if(typeof value==='number')unchanged=Number(current)===value;
     else unchanged=String(current??'')===String(value);
     if(!unchanged)changes[key]=value;
    });
    if(!Object.keys(changes).length){message.textContent='No changes to save.';return;}
    save.disabled=true;message.textContent='Saving settings…';
    try{
     const response=await apiFetch('/admin/configuration',{method:'PATCH',body:JSON.stringify({values:changes})}),result=await response.json();
     if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'Unable to save these settings.');
     const refreshed=await AdminPages.api('/admin/configuration');values=refreshed.values||result.values||values;
     toast(result.message||'Settings saved.');renderDetail(id);
    }catch(error){message.textContent=error.message||'Unable to save these settings.';}
    finally{save.disabled=false;}
   };
   const reset=form.querySelector('[data-reset-settings]');
   reset.onclick=async()=>{
    if(!confirm('Restore deployment defaults for all editable AI, retrieval, and application settings?'))return;
    reset.disabled=true;message.textContent='Restoring defaults…';
    try{
     const response=await apiFetch('/admin/configuration',{method:'DELETE'}),result=await response.json();
     if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'Unable to restore settings.');
     const refreshed=await AdminPages.api('/admin/configuration');values=refreshed.values||result.values||values;
     toast(result.message||'Deployment defaults restored.');renderDetail(id);
    }catch(error){message.textContent=error.message||'Unable to restore settings.';}
    finally{reset.disabled=false;}
   };
  };
  host.querySelectorAll('[data-setting]').forEach(button=>button.onclick=()=>renderDetail(button.dataset.setting));
 }};
});
settingsPage.querySelector('form').remove();
