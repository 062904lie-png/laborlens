/* Reparents existing upload fields; uploadFile and all upload APIs are reused. */
window.AdminUpload=(()=>{
 const $=id=>document.getElementById(id),input=$('file-input'),modal=input.closest('.kb-upload-modal'),body=input.closest('.panel-body');
 const groups=[...body.querySelectorAll('.form-group')],zone=body.querySelector('.upload-zone'),suggest=$('kb-category-suggestion').parentElement,msg=$('upload-msg');
 const oldProcess=[...body.querySelectorAll('button')].find(b=>b.textContent==='Upload and Process');oldProcess?.remove();
 const fields=new Map(groups.map(g=>[g.querySelector('input,select,.kb-secondary-grid')?.id,g]));
 const typeInput=$('kb-document-type');
 if(typeInput?.tagName==='INPUT'){
  const select=document.createElement('select');select.id='kb-document-type';select.required=true;select.setAttribute('aria-label','Document Type');
  const existing=typeInput.value;select.innerHTML='<option value="">Select type</option><option>PDF</option><option>DOCX</option><option>IMAGE</option><option>Other</option>';select.value=['PDF','DOCX','IMAGE','Other'].includes(String(existing).toUpperCase())?String(existing).toUpperCase():'';
  typeInput.replaceWith(select);fields.set('kb-document-type',select.closest('.form-group'));
 }
 ['kb-date-issued','kb-effective-date'].forEach(id=>{const field=$(id);if(field){field.type='date';field.placeholder='';}});
 const labels={'kb-title':'Document Title *','kb-document-type':'Document Type *','kb-category':'Main Category *','kb-issuing-agency':'Issued By','kb-source':'Source Name','kb-law-number':'Law / Order Number','kb-source-url':'Source URL','kb-date-issued':'Issued Date','kb-effective-date':'Date the Rule Takes Effect'};
 Object.entries(labels).forEach(([id,text])=>{const field=$(id),label=field?.closest('.form-group')?.querySelector('label');if(field&&['kb-title','kb-category','kb-document-type'].includes(id))field.required=true;if(label)label.textContent=text;});
 const basic=['kb-title','kb-document-type','kb-category','kb-issuing-agency','kb-source','kb-law-number','kb-source-url','kb-date-issued','kb-effective-date'];
 const wizard=document.createElement('div');wizard.className='upload-wizard';
 wizard.innerHTML='<div class="upload-steps add-document-stepper" aria-label="Upload progress"><span data-step-indicator="0"><b>1</b><em>Upload File</em></span><span data-step-indicator="1"><b>2</b><em>Basic Information</em></span><span data-step-indicator="2"><b>3</b><em>Review &amp; Process</em></span></div><section class="upload-step add-document-upload" data-step="0"></section><section class="upload-step add-document-form" data-step="1" hidden><div id="upload-validation" class="upload-validation" role="alert" hidden></div><div class="upload-fields upload-basic-grid"></div><details class="admin-technical upload-advanced"><summary>More Document Details <small>(Optional)</small></summary><p>Add additional legal metadata such as extra categories, legal role, articles, implementing issuances, and more.</p><div class="upload-fields"></div></details></section><section class="upload-step add-document-review" data-step="2" hidden><div class="upload-review-note"><b>ⓘ Please review the document details below before processing.</b><span>Once processed, the document will be indexed and made available for retrieval.</span></div><div id="upload-summary"></div></section><div class="quick-actions"><button type="button" class="btn" id="upload-back">Cancel</button><button type="button" class="btn btn-primary" id="upload-next">Next →</button></div>';
 const steps=[...wizard.querySelectorAll('[data-step]')],grids=steps[1].querySelectorAll('.upload-fields');
 steps[0].append(zone,input);input.className='upload-native-input';input.setAttribute('aria-label','Choose legal document');steps[0].insertAdjacentHTML('beforeend','<div class="upload-selected-file" id="upload-selected-file" hidden></div><aside class="upload-supported-info"><b>ⓘ Supported file types</b><span>PDF, DOCX, and scanned images. Documents are automatically processed, with text extracted and indexed for the AI assistant.</span></aside>');
 zone.innerHTML='<span class="upload-cloud" aria-hidden="true">⇧</span><strong>Drag and drop your document here</strong><span>or</span><button type="button" class="btn btn-primary" id="upload-choose-file">▧&nbsp; Choose File</button><small>PDF, DOCX, PNG, JPG, TIFF, BMP, or WEBP · Max 50 MB</small>';
 wizard.querySelector('#upload-choose-file').onclick=event=>{event.stopPropagation();input.click();};
 basic.forEach(id=>{if(fields.has(id))grids[0].append(fields.get(id));});
 groups.filter(g=>!basic.includes(g.querySelector('input,select,.kb-secondary-grid')?.id)).forEach(g=>grids[1].append(g));
 grids[1].append(suggest);grids[1].insertAdjacentHTML('beforeend','<p>One main category and up to three extra categories. Version status controls eligibility for retrieval; use document metadata to change active status after upload.</p>');
 body.replaceChildren(wizard,msg);
 let step=0,busy=false;
 function show(n){step=n;steps.forEach((el,i)=>el.hidden=i!==n);wizard.querySelectorAll('[data-step-indicator]').forEach((el,i)=>{el.classList.toggle('active',i===n);el.classList.toggle('complete',i<n);el.querySelector('b').textContent=i<n?'✓':i+1;});$('upload-back').disabled=busy;$('upload-back').textContent=n===0?'Cancel':'← Back';$('upload-next').disabled=busy;$('upload-next').textContent=n===2?'⚙ Process Document':'Next →';}
 function fileSize(size){return size>=1048576?`${(size/1048576).toFixed(1)} MB`:`${Math.ceil(size/1024)} KB`;}
 function fileChanged(){const file=input.files[0],card=$('upload-selected-file');card.hidden=!file;if(!file){card.replaceChildren();return;}card.innerHTML=`<span class="upload-file-icon">${file.name.toLowerCase().endsWith('.pdf')?'PDF':'FILE'}</span><strong>${esc(file.name)}<small>${fileSize(file.size)}</small></strong><b aria-label="File selected">✓</b><button type="button" aria-label="Remove selected file">×</button>`;card.querySelector('button').onclick=()=>{input.value='';fileChanged();};}
 input.onchange=fileChanged;
 zone.ondragover=e=>e.preventDefault();
 zone.ondrop=e=>{e.preventDefault();if(!busy&&e.dataTransfer.files.length){input.files=e.dataTransfer.files;fileChanged();}};
 $('upload-back').onclick=()=>{if(busy)return;if(step===0){modal?.classList.remove('show');return;}show(step-1);};
 $('upload-next').onclick=async()=>{
  if(busy)return;
  if(step===0){if(!input.files[0])return toast('Choose a file first.',true);if(input.files[0].size>50*1024*1024)return toast('Maximum file size is 50 MB.',true);return show(1);}
  if(step===1){
   const invalid=[...grids[0].querySelectorAll('input,select')].find(el=>!el.checkValidity());if(invalid){const message=$('upload-validation');message.hidden=false;message.textContent=`Please complete: ${invalid.closest('.form-group')?.querySelector('label')?.textContent.replace(' *','')||'required field'}.`;invalid.focus();return invalid.reportValidity();}$('upload-validation').hidden=true;
   if(getSelectedSecondaryCategories('kb-secondary-categories').length>3)return toast('Choose at most three extra categories.',true);
   const value=id=>esc($(id)?.value?.trim()||'Not provided'),formatDate=id=>{const raw=$(id)?.value;if(!raw)return 'Not provided';const date=new Date(raw+'T00:00:00');return Number.isNaN(date)?esc(raw):esc(date.toLocaleDateString(undefined,{month:'long',day:'numeric',year:'numeric'}));},file=input.files[0];$('upload-summary').innerHTML=`<article class="upload-review-card"><div class="upload-review-file"><span>${file.name.toLowerCase().endsWith('.pdf')?'PDF':'FILE'}</span><strong>${esc(file.name)}<small>${fileSize(file.size)}</small></strong></div>${[['Document Title',value('kb-title')],['Document Type',value('kb-document-type')],['Main Category',value('kb-category')],['Issued By',value('kb-issuing-agency')],['Source Name',value('kb-source')],['Source URL',value('kb-source-url')],['Law / Order Number',value('kb-law-number')],['Issued Date',formatDate('kb-date-issued')],['Date the Rule Takes Effect',formatDate('kb-effective-date')]].map(([label,value])=>`<div class="upload-review-row"><span>${label}</span><strong>${value}</strong></div>`).join('')}</article>`;return show(2);
  }
  busy=true;show(2);wizard.querySelectorAll('input,select,button').forEach(el=>el.disabled=true);
  try{const result=await uploadFile(input);if(result){fileChanged();$('upload-summary').replaceChildren();step=0;}}finally{busy=false;wizard.querySelectorAll('input,select,button').forEach(el=>el.disabled=false);syncSecondaryCategoryOptions('kb-secondary-categories',$('kb-category').value);show(step);}
 };
 show(0);
 return {open(){modal?.classList.add('show');setTimeout(()=>input.focus(),0);}};
})();
