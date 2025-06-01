// Validación de formulario
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            const inputs = form.querySelectorAll('input[type="number"]');
            let isValid = true;

            inputs.forEach(input => {
                if (input.value === '') {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                event.preventDefault();
                alert('Por favor, complete todos los campos.');
            }
        });
    }
}); 