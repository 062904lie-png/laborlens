/* Knowledge Base category browser. Categories are derived from the current document library. */
const categoryDescriptions = [
  'Labor Code and general employment rights.',
  'Wages, overtime, compensation and statutory benefits.',
  'Leave entitlements and eligibility.',
  'Household workers and Kasambahay references.',
  'Workplace harassment and discrimination protections.',
  'Social insurance and related benefits.',
  'Workplace health, safety and prevention.',
  'Official DOLE guidance and resources.',
  'Unions, collective bargaining and labor relations.',
  'Court decisions and legal interpretation.',
  'Labor advisories and official notices.',
  'Other legal references.',
  'CSC rules, appointments, administrative cases and government leave policies. Government-employment sources; verify applicability separately from private-sector labor rules.'
];

const categoryPage = AdminPages.register(
  'categories',
  'Categories',
  'Manage the legal categories used for document classification and retrieval.',
  async (search, page) => {
    const U = AdminUI;
    const documents = await AdminPages.api('/kb/documents');
    const lastUpdated = rows => rows
      .map(document => document.updated_at || document.last_indexed_at || document.created_at)
      .filter(Boolean)
      .sort((a, b) => new Date(b) - new Date(a))[0];
    const categories = KB_CATEGORIES.map((name, index) => ({
      name,
      description: categoryDescriptions[index] || '',
      documents: documents.filter(document => new Set([
        document.primary_category || document.category,
        ...parseMetadataList(document.secondary_categories)
      ]).has(name))
    }));
    const query = search.trim().toLowerCase();
    const filtered = categories.filter(category =>
      `${category.name} ${category.description}`.toLowerCase().includes(query)
    );
    const pageSize = adminPageSize();
    const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
    const currentPage = Math.min(Math.max(Number(page.dataset.offset || 0), 0), pageCount - 1);
    const start = currentPage * pageSize;
    const visible = filtered.slice(start, start + pageSize);
    const totalLastUpdated = lastUpdated(documents);
    const summary = [
      ['folder', 'Total Categories', categories.length, 'Legal classifications in the system'],
      ['file', 'Total Documents', documents.length, 'Documents assigned to these categories'],
      ['clock', 'Last Document Update', dateLabel(totalLastUpdated), totalLastUpdated ? 'Most recent library update' : 'No document update recorded']
    ];
    const pageButtons = Array.from({ length: pageCount }, (_, index) =>
      `<button class="${index === currentPage ? 'active' : ''}" type="button" data-category-page="${index}" aria-label="Page ${index + 1}" aria-current="${index === currentPage ? 'page' : 'false'}">${index + 1}</button>`
    ).join('');
    const rows = visible.length ? visible.map(category => {
      const latest = dateLabel(lastUpdated(category.documents));
      return `<tr>
        <td data-label="#">${KB_CATEGORIES.indexOf(category.name) + 1}</td>
        <td data-label="Category Name"><strong>${esc(category.name)}</strong></td>
        <td data-label="Description">${esc(category.description)}</td>
        <td data-label="Documents"><span class="kb-category-document-count">${category.documents.length}</span></td>
        <td data-label="Last Updated">${esc(latest)}</td>
        <td data-label="Action"><button class="btn btn-sm" type="button" data-category="${esc(category.name)}">View</button></td>
      </tr>`;
    }).join('') : '<tr><td colspan="6" class="empty">No matching categories.</td></tr>';

    return {
      html: `<section class="kb-category-summary" aria-label="Category summary">${summary.map(([icon, label, value, note]) => `
        <article>
          <span aria-hidden="true">${U.icon(icon)}</span>
          <div><small>${esc(label)}</small><strong>${esc(value)}</strong><p>${esc(note)}</p></div>
        </article>`).join('')}</section>
        <section class="kb-category-table-card" aria-label="Knowledge base categories">
          <div class="tbl-wrap"><table class="kb-category-table">
            <thead><tr><th>#</th><th>Category Name</th><th>Description</th><th>Documents</th><th>Last Updated</th><th>Action</th></tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
          <footer class="kb-category-pager">
            <span>Showing ${filtered.length ? start + 1 : 0}&ndash;${Math.min(start + pageSize, filtered.length)} of ${filtered.length} categories</span>
            <nav aria-label="Categories pagination">
              <button type="button" data-category-page="${currentPage - 1}" ${currentPage === 0 ? 'disabled' : ''}>‹ Previous</button>
              ${pageButtons}
              <button type="button" data-category-page="${currentPage + 1}" ${currentPage >= pageCount - 1 ? 'disabled' : ''}>Next ›</button>
            </nav>
          </footer>
        </section>`,
      mount(host) {
        host.querySelectorAll('[data-category-page]').forEach(button => {
          button.onclick = () => {
            if (button.disabled) return;
            page.dataset.offset = button.dataset.categoryPage;
            AdminPages.open('categories');
          };
        });
        host.querySelectorAll('[data-category]').forEach(button => {
          button.onclick = () => {
            const category = categories.find(item => item.name === button.dataset.category);
            if (!category) return;
            const current = category.documents.filter(document => ['current', 'active'].includes(document.version_status)).length;
            U.drawer(
              'Category Details',
              `<h3>${esc(category.name)}</h3>${U.badge('System category', 'green')}<p>${esc(category.description)}</p>` +
              U.table(['Field', 'Value'], [
                ['Documents', category.documents.length],
                ['Current', current],
                ['Last document update', esc(dateLabel(lastUpdated(category.documents)))],
                ['Category updated by', 'Not recorded']
              ]),
              '<button class="btn btn-primary" id="category-documents">View Documents</button>'
            );
            document.getElementById('category-documents').onclick = () => {
              closeDocModal();
              U.go('knowledge');
              AdminKnowledge.viewCategory(category.name);
            };
          };
        });
      }
    };
  }
);

categoryPage.classList.add('kb-categories-page');
categoryPage.querySelector('.page-hdr h1')?.remove();
const categoryFilter = categoryPage.querySelector('.workbench-filters');
categoryFilter.classList.add('kb-category-filters');
categoryFilter.innerHTML = `${AdminUI.icon('search')}<label class="kb-category-search"><span class="sr-only">Search categories</span><input type="search" maxlength="200" placeholder="Search categories..."/></label><button class="btn btn-primary" type="submit">${AdminUI.icon('search')} Search / Filter</button><button class="btn" type="reset">Reset</button>`;
