document.getElementById('model').addEventListener('change', function() {
    var algorithmFields = document.getElementById('algorithm-fields');
    algorithmFields.style.display = this.value ? 'block' : 'none';
    document.getElementById('main-params').style.display = 'none';
});

document.getElementById('algorithm').addEventListener('change', function() {
    var mainParams = document.getElementById('main-params');
    var model = document.getElementById('model').value;
    var algorithm = this.value;

    document.getElementById('main-params-fpgm-l1-l2').style.display = 'none';
    document.getElementById('main-params-snip-synflow-our_algo').style.display = 'none';

    if ((algorithm === 'FPGM' || algorithm === 'L1' || algorithm === 'L2') && model === 'detr') {
        document.getElementById('main-params-fpgm-l1-l2').style.display = 'block';
        document.getElementById('rate_dist_field').style.display = algorithm === 'FPGM' ? 'block' : 'none';
        document.getElementById('rate_norm_field').style.display = algorithm !== 'FPGM' ? 'block' : 'none';
    } else if (algorithm === 'SNIP' || algorithm === 'Synflow' || algorithm === 'our_algo') {
        document.getElementById('main-params-snip-synflow-our_algo').style.display = 'block';
        document.getElementById('pruner').value = algorithm.toLowerCase();
    }

    mainParams.style.display = algorithm ? 'block' : 'none';
});

const form = document.getElementById('training-form');
form.addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const params = new URLSearchParams(formData).toString();
    const eventSource = new EventSource(`/stream_output?${params}`);
    const outputDiv = document.getElementById('output');
    const trainingImage = document.getElementById('training-image');
    const outputDir = formData.get('output_dir');
    trainingImage.src = `/images/${outputDir}/training_plots.png`;

    outputDiv.innerHTML = '';
    eventSource.onmessage = function(event) {
        if (event.data.includes('Process completed')) {
            eventSource.close();
        }
        const lines = outputDiv.innerHTML.split('<br>');
        if (lines.length > 100) {
            lines.splice(0, lines.length - 100); // Keep only the last 100 lines
        }
        outputDiv.innerHTML = lines.join('<br>') + '<br>' + event.data;
        outputDiv.scrollTop = outputDiv.scrollHeight; // Auto-scroll to the bottom
    };

    function refreshImage() {
        const timestamp = new Date().getTime();
        const url = trainingImage.src.split('?')[0];
        trainingImage.src = `${url}?timestamp=${timestamp}`;
    }
    setInterval(refreshImage, 5000); // Refresh image every 5 seconds
});
