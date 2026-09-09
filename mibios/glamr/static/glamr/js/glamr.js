/* Filter form: on submission, disable blank input elements so that they don't
 * clutter the URL query string
 */
for (const i of document.forms) {
    if (i.className == 'filter') {
        i.addEventListener('submit', event => {
            const form = event.target;
            for (const item of Array.from(form.elements)) {
                if (item.localName == 'input' && !item.value) {
                    item.disabled = true;
                } else if (item.localName == 'select' && !item.value) {
                    item.disabled = true;
                }
            }
        });
    }
}
