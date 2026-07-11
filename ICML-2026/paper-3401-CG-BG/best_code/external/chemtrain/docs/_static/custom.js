document.addEventListener("DOMContentLoaded", function() {
    // Define the function to replace <span> elements with the specified class
    function replaceSpanWithClass(className) {
        // Find the <code> section with the specified class
        var codeSections = document.querySelectorAll('code.' + className);

        codeSections.forEach(function(codeSection) {

            // Find all <span> elements within the <code> section
            var spans = codeSection.querySelectorAll('span');

            // Initialize an empty string to store the replaced text
            var replacedText = "";

            // Iterate over the found <span> elements
            spans.forEach(function (span) {
                // Extract the content inside the <span> tag
                var content = span.textContent;

                // Split the content string using the '.' character
                var parts = content.split(/(?=\.)/);

                // Create span elements for each part and join them together
                var spanElements = parts.map(function (part) {
                    return '<span>' + part + '</span>';
                }).join('');

                // Add the span elements to the replaced text
                replacedText += spanElements;
            });

            // Set the new content
            codeSection.innerHTML = replacedText;

        });
    }

    // Call the function with the specified class name
    replaceSpanWithClass('modulename');

});
