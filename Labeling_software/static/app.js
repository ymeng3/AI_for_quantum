// Global state
let images = [];
let labels = {};
let currentImage = null;
let currentLabels = {
    reconstruction: [],  // List of selected reconstruction types
    reconstruction_scores: {},  // Dict: {reconstruction: quality_score (0-10)}
    notes: ''  // Optional notes/metadata
};
let currentLabelerName = '';
// Pagination state
let currentPage = 1;
const imagesPerPage = 100; // Show 100 images per page

// Pairwise comparison state
let currentMode = 'absolute'; // 'absolute' or 'pairwise'
let pairwiseImage1 = null;
let pairwiseImage2 = null;
let pairwiseComparisons = {}; // {reconstruction_type: winner} where winner is '1', '2', or 'tie'
let pairwiseLabelerName = '';
let pairwiseNotes = '';

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadImages();
    loadLabels();
    setupEventListeners();
    initializePairwiseMode();
});

// Setup event listeners
function setupEventListeners() {
    // Pagination buttons
    document.getElementById('prevPageBtn').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderImageGrid(document.getElementById('filterSelect').value);
        }
    });
    document.getElementById('nextPageBtn').addEventListener('click', () => {
        currentPage++;
        renderImageGrid(document.getElementById('filterSelect').value);
    });
    
    // Labeler name input
    document.getElementById('labelerName').addEventListener('input', (e) => {
        currentLabelerName = e.target.value.trim();
    });
    
    // Notes input
    document.getElementById('notesInput').addEventListener('input', (e) => {
        currentLabels.notes = e.target.value;
    });
    
    // Brightness slider
    const brightnessSlider = document.getElementById('brightnessSlider');
    const brightnessValue = document.getElementById('brightnessValue');
    const mainImage = document.getElementById('mainImage');
    
    brightnessSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        brightnessValue.textContent = `${value}%`;
        mainImage.style.filter = `brightness(${value}%)`;
    });
    
    // Reconstruction checkboxes (multiple select)
    document.querySelectorAll('.reconstruction-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const value = e.target.dataset.value;
            const qualityControl = document.querySelector(`.reconstruction-quality-control[data-recon="${value}"]`);
            
            if (e.target.checked) {
                // Add to list and show quality score slider
                if (!currentLabels.reconstruction.includes(value)) {
                    currentLabels.reconstruction.push(value);
                }
                // Initialize default score of 5 if not set
                if (!currentLabels.reconstruction_scores[value]) {
                    currentLabels.reconstruction_scores[value] = 5;
                }
                if (qualityControl) {
                    qualityControl.style.display = 'block';
                    // Set slider value
                    const slider = qualityControl.querySelector('.reconstruction-quality-slider');
                    const valueSpan = qualityControl.querySelector('.quality-score-value');
                    if (slider && valueSpan) {
                        slider.value = currentLabels.reconstruction_scores[value];
                        valueSpan.textContent = currentLabels.reconstruction_scores[value];
                    }
                }
            } else {
                // Remove from list and hide quality control
                currentLabels.reconstruction = currentLabels.reconstruction.filter(r => r !== value);
                delete currentLabels.reconstruction_scores[value];
                if (qualityControl) {
                    qualityControl.style.display = 'none';
                    // Reset slider to default
                    const slider = qualityControl.querySelector('.reconstruction-quality-slider');
                    const valueSpan = qualityControl.querySelector('.quality-score-value');
                    if (slider && valueSpan) {
                        slider.value = 5;
                        valueSpan.textContent = '5';
                    }
                }
            }
        });
    });
    
    // Reconstruction quality score sliders
    document.querySelectorAll('.reconstruction-quality-slider').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const recon = e.target.dataset.recon;
            const score = parseInt(e.target.value);
            const valueSpan = document.querySelector(`.quality-score-value[data-recon="${recon}"]`);
            
            if (!isNaN(score) && score >= 0 && score <= 10) {
                currentLabels.reconstruction_scores[recon] = score;
                if (valueSpan) {
                    valueSpan.textContent = score;
                }
            }
        });
    });
    
    // Save button
    document.getElementById('saveBtn').addEventListener('click', saveLabel);
    
    // Clear button
    document.getElementById('clearBtn').addEventListener('click', clearLabels);
    
    // Mode selector buttons
    document.getElementById('absoluteModeBtn').addEventListener('click', () => switchMode('absolute'));
    document.getElementById('pairwiseModeBtn').addEventListener('click', () => switchMode('pairwise'));
    
    // Filter select - resets to page 1 when changed
    document.getElementById('filterSelect').addEventListener('change', (e) => {
        currentPage = 1;
        renderImageGrid(e.target.value);
    });
    
    // Export button
    document.getElementById('exportBtn').addEventListener('click', exportLabels);
}

// Load images from API
async function loadImages() {
    try {
        const response = await fetch('/api/images');
        images = await response.json();
        console.log(`Loaded ${images.length} images`);
        if (images.length > 0) {
            console.log('First image path:', images[0].path);
        }
        renderImageGrid();
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

// Load labels from API
async function loadLabels() {
    try {
        const response = await fetch('/api/labels');
        const labelsData = await response.json();
        
        // Convert to object for easy lookup
        labels = {};
        labelsData.forEach(label => {
            labels[label.file_path] = label;
        });
        
        renderLabelsTable();
        updateImageGridStatus();
    } catch (error) {
        console.error('Error loading labels:', error);
    }
}

// Render image grid
function renderImageGrid(filter = 'all') {
    const grid = document.getElementById('imageGrid');
    grid.innerHTML = '';
    
    let filteredImages = images;
    if (filter === 'labeled') {
        filteredImages = images.filter(img => {
            const label = labels[img.path];
            return label && (label.quality || label.reconstruction);
        });
    } else if (filter === 'unlabeled') {
        filteredImages = images.filter(img => {
            const label = labels[img.path];
            return !label || (!label.quality && !label.reconstruction);
        });
    }
    
    // Pagination
    const totalPages = Math.ceil(filteredImages.length / imagesPerPage);
    const startIndex = (currentPage - 1) * imagesPerPage;
    const endIndex = startIndex + imagesPerPage;
    const displayImages = filteredImages.slice(startIndex, endIndex);
    
    // Update count display
    const countEl = document.getElementById('imageCount');
    if (countEl) {
        countEl.textContent = `Showing ${startIndex + 1}-${Math.min(endIndex, filteredImages.length)} of ${filteredImages.length}`;
    }
    
    // Update pagination controls
    const paginationControls = document.getElementById('paginationControls');
    const pageInfo = document.getElementById('pageInfo');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    
    if (totalPages > 1) {
        paginationControls.style.display = 'block';
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages;
    } else {
        paginationControls.style.display = 'none';
    }
    
    console.log(`Rendering page ${currentPage}/${totalPages}: ${displayImages.length} images (out of ${filteredImages.length} total)`);
    
    displayImages.forEach((img, index) => {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.dataset.path = img.path;
        
        const imgEl = document.createElement('img');
        // Properly encode the path - split by / and encode each segment
        const encodedPath = img.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
        const imageUrl = `/api/images/${encodedPath}`;
        imgEl.src = imageUrl;
        imgEl.alt = img.name;
        imgEl.loading = 'lazy';
        imgEl.style.width = '100%';
        imgEl.style.height = '100%';
        imgEl.style.objectFit = 'contain';
        imgEl.onerror = function() {
            console.error('Failed to load image:', img.path, 'URL:', imageUrl);
            this.style.border = '2px solid red';
            this.alt = 'Failed to load: ' + img.name;
            // Show error text
            const errorDiv = document.createElement('div');
            errorDiv.textContent = 'Error';
            errorDiv.style.color = 'red';
            errorDiv.style.fontSize = '10px';
            errorDiv.style.position = 'absolute';
            errorDiv.style.top = '50%';
            errorDiv.style.left = '50%';
            errorDiv.style.transform = 'translate(-50%, -50%)';
            item.appendChild(errorDiv);
        };
        imgEl.onload = function() {
            console.log('Successfully loaded image:', img.name);
        };
        
        const badge = document.createElement('div');
        badge.className = 'status-badge';
        const label = labels[img.path];
        if (label && label.quality && label.reconstruction) {
            badge.textContent = '✓';
            badge.classList.add('labeled');
        } else if (label && (label.quality || label.reconstruction)) {
            badge.textContent = '~';
            badge.classList.add('partial');
        }
        
        item.appendChild(imgEl);
        if (badge.textContent) {
            item.appendChild(badge);
        }
        
        item.addEventListener('click', () => selectImage(img));
        grid.appendChild(item);
    });
}

// Filter images
function filterImages(filter) {
    renderImageGrid(filter);
}

// Update image grid status badges
function updateImageGridStatus() {
    document.querySelectorAll('.image-item').forEach(item => {
        const path = item.dataset.path;
        const label = labels[path];
        let badge = item.querySelector('.status-badge');
        
        // Check if image is labeled (has reconstruction selected)
        if (label && label.reconstruction) {
            try {
                const recon = typeof label.reconstruction === 'string' ? JSON.parse(label.reconstruction) : label.reconstruction;
                if (Array.isArray(recon) && recon.length > 0) {
                    if (!badge) {
                        badge = document.createElement('div');
                        badge.className = 'status-badge';
                        item.appendChild(badge);
                    }
                    badge.textContent = '✓';
                    badge.className = 'status-badge labeled';
                } else if (recon) {
                    if (!badge) {
                        badge = document.createElement('div');
                        badge.className = 'status-badge';
                        item.appendChild(badge);
                    }
                    badge.textContent = '✓';
                    badge.className = 'status-badge labeled';
                } else if (badge) {
                    badge.remove();
                }
            } catch {
                if (badge) {
                    badge.remove();
                }
            }
        } else if (badge) {
            badge.remove();
        }
    });
}

// Select an image
async function selectImage(img) {
    currentImage = img;
    
    // Update main image display
    const mainImg = document.getElementById('mainImage');
    // Properly encode the path - split by / and encode each segment
    const encodedPath = img.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
    mainImg.src = `/api/images/${encodedPath}`;
    mainImg.onerror = function() {
        console.error('Failed to load image:', img.path, 'URL:', this.src);
        this.style.border = '2px solid red';
    };
    mainImg.onload = function() {
        this.style.border = '2px solid #333';
        // Apply current brightness setting
        const brightnessSlider = document.getElementById('brightnessSlider');
        const brightness = parseInt(brightnessSlider.value);
        this.style.filter = `brightness(${brightness}%)`;
    };
    document.getElementById('imageInfo').textContent = img.name;
    
    // Highlight selected image in grid
    document.querySelectorAll('.image-item').forEach(item => {
        item.classList.remove('selected');
    });
    document.querySelector(`.image-item[data-path="${img.path}"]`)?.classList.add('selected');
    
    // Load existing labels
    try {
        const response = await fetch(`/api/labels/${encodeURIComponent(img.path)}`);
        const labelData = await response.json();
        
        // Handle reconstruction (can be array or single value)
        if (labelData.reconstruction) {
            if (Array.isArray(labelData.reconstruction)) {
                currentLabels.reconstruction = labelData.reconstruction;
            } else {
                currentLabels.reconstruction = [labelData.reconstruction];
            }
        } else {
            currentLabels.reconstruction = [];
        }
        
        // Handle reconstruction scores (quality scores per reconstruction)
        if (labelData.reconstruction_scores) {
            if (typeof labelData.reconstruction_scores === 'object') {
                currentLabels.reconstruction_scores = labelData.reconstruction_scores;
            } else {
                try {
                    currentLabels.reconstruction_scores = JSON.parse(labelData.reconstruction_scores);
                } catch {
                    currentLabels.reconstruction_scores = {};
                }
            }
        } else {
            currentLabels.reconstruction_scores = {};
        }
        
        // Load labeler name
        if (labelData.labeler_name) {
            currentLabelerName = labelData.labeler_name;
            document.getElementById('labelerName').value = currentLabelerName;
        }
        
        // Load notes
        if (labelData.notes) {
            currentLabels.notes = labelData.notes;
            document.getElementById('notesInput').value = currentLabels.notes;
        } else {
            currentLabels.notes = '';
            document.getElementById('notesInput').value = '';
        }
        
        // Update button states
        updateButtonStates();
    } catch (error) {
        console.error('Error loading label:', error);
        currentLabels.reconstruction = [];
        currentLabels.reconstruction_scores = {};
        currentLabels.notes = '';
        document.getElementById('notesInput').value = '';
        updateButtonStates();
    }
    
    // Reset brightness slider when selecting new image
    const brightnessSlider = document.getElementById('brightnessSlider');
    brightnessSlider.value = 100;
    document.getElementById('brightnessValue').textContent = '100%';
    mainImage.style.filter = 'brightness(100%)';
}

// Update button states based on current labels
function updateButtonStates() {
    // Update reconstruction checkboxes and quality sliders
    document.querySelectorAll('.reconstruction-checkbox').forEach(checkbox => {
        const value = checkbox.dataset.value;
        checkbox.checked = currentLabels.reconstruction.includes(value);
        
        // Show/hide quality control
        const qualityControl = document.querySelector(`.reconstruction-quality-control[data-recon="${value}"]`);
        if (qualityControl) {
            if (checkbox.checked) {
                qualityControl.style.display = 'block';
                // Set slider and value display
                const slider = qualityControl.querySelector('.reconstruction-quality-slider');
                const valueSpan = qualityControl.querySelector('.quality-score-value');
                if (slider && valueSpan) {
                    const score = currentLabels.reconstruction_scores[value] || 5;
                    slider.value = score;
                    valueSpan.textContent = score;
                }
            } else {
                qualityControl.style.display = 'none';
            }
        }
    });
}

// Save label
async function saveLabel() {
    if (!currentImage) {
        alert('Please select an image first');
        return;
    }
    
    if (!currentLabelerName) {
        alert('Please enter your name before saving');
        return;
    }
    
    try {
        const response = await fetch('/api/labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: currentImage.path,
                file_name: currentImage.name,
                quality: null,  // No longer used, but kept for backward compatibility
                reconstruction: currentLabels.reconstruction.length > 0 ? currentLabels.reconstruction : null,
                reconstruction_scores: Object.keys(currentLabels.reconstruction_scores).length > 0 ? currentLabels.reconstruction_scores : null,
                labeler_name: currentLabelerName,
                notes: currentLabels.notes || null
            })
        });
        
        if (response.ok) {
            await loadLabels(); // Reload labels to update table and grid
            alert('Label saved successfully!');
        } else {
            const error = await response.json();
            alert('Error saving label: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error saving label:', error);
        alert('Error saving label');
    }
}

// Clear labels
function clearLabels() {
    currentLabels.reconstruction = [];
    currentLabels.reconstruction_scores = {};
    currentLabels.notes = '';
    document.getElementById('notesInput').value = '';
    updateButtonStates();
}

// Render labels table
function renderLabelsTable() {
    const tbody = document.getElementById('labelsTableBody');
    tbody.innerHTML = '';
    
    const sortedLabels = Object.values(labels).sort((a, b) => {
        return new Date(b.updated_at) - new Date(a.updated_at);
    });
    
    sortedLabels.forEach(label => {
        const row = document.createElement('tr');
        row.dataset.path = label.file_path;
        
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.className = 'delete-btn';
        deleteBtn.onclick = (e) => {
            e.stopPropagation(); // Prevent row click
            deleteLabel(label.file_path);
        };
        
        // Parse reconstruction (can be array or single value)
        let reconstructionText = '-';
        if (label.reconstruction) {
            try {
                const recon = typeof label.reconstruction === 'string' ? JSON.parse(label.reconstruction) : label.reconstruction;
                if (Array.isArray(recon)) {
                    reconstructionText = recon.join(', ');
                } else {
                    reconstructionText = recon;
                }
            } catch {
                reconstructionText = label.reconstruction;
            }
        }
        
        // Parse scores
        let scoresText = '-';
        if (label.reconstruction_scores) {
            try {
                const scores = typeof label.reconstruction_scores === 'string' ? JSON.parse(label.reconstruction_scores) : label.reconstruction_scores;
                if (typeof scores === 'object' && scores !== null) {
                    scoresText = Object.entries(scores).map(([k, v]) => `${k}: ${v}`).join(', ');
                }
            } catch {
                scoresText = '-';
            }
        }
        
        const notesText = label.notes || '-';
        const notesDisplay = notesText.length > 50 ? notesText.substring(0, 50) + '...' : notesText;
        
        row.innerHTML = `
            <td>${label.file_name}</td>
            <td>${reconstructionText}</td>
            <td>${scoresText}</td>
            <td>${label.labeler_name || '-'}</td>
            <td title="${notesText !== '-' ? notesText : ''}" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${notesDisplay}</td>
            <td></td>
        `;
        
        // Add delete button to the last cell
        const actionCell = row.querySelector('td:last-child');
        actionCell.appendChild(deleteBtn);
        
        row.addEventListener('click', (e) => {
            // Don't select image if clicking on delete button
            if (e.target.classList.contains('delete-btn')) {
                return;
            }
            const img = images.find(i => i.path === label.file_path);
            if (img) {
                selectImage(img);
            }
        });
        
        tbody.appendChild(row);
    });
}

// Delete a label
async function deleteLabel(filePath) {
    if (!confirm(`Are you sure you want to delete the label for "${filePath.split('/').pop()}"?`)) {
        return;
    }
    
    try {
        const encodedPath = encodeURIComponent(filePath);
        const response = await fetch(`/api/labels/${encodedPath}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            // Remove from local state
            delete labels[filePath];
            
            // Reload labels table and update image grid
            renderLabelsTable();
            updateImageGridStatus();
            
            // Clear main image if it was the deleted one
            if (currentImage && currentImage.path === filePath) {
                currentImage = null;
                document.getElementById('mainImage').src = '';
                document.getElementById('imageInfo').textContent = 'Select an image to label';
                clearLabels();
            }
            
            alert('Label deleted successfully!');
        } else {
            alert('Error deleting label: ' + (result.message || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting label:', error);
        alert('Error deleting label');
    }
}

// Export labels as CSV
function exportLabels() {
    if (currentMode === 'pairwise') {
        window.location.href = '/api/pairwise/export';
    } else {
        window.location.href = '/api/labels/export';
    }
}

// Switch between absolute and pairwise modes
function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.getElementById('absoluteModeBtn').classList.toggle('active', mode === 'absolute');
    document.getElementById('pairwiseModeBtn').classList.toggle('active', mode === 'pairwise');
    
    // Show/hide mode sections
    document.getElementById('absoluteMode').style.display = mode === 'absolute' ? 'flex' : 'none';
    document.getElementById('pairwiseMode').style.display = mode === 'pairwise' ? 'block' : 'none';
    
    if (mode === 'pairwise') {
        // Initialize pairwise mode - load random pair
        loadRandomPair();
    }
}

// Initialize pairwise mode
function initializePairwiseMode() {
    // Pairwise labeler name input
    document.getElementById('pairwiseLabelerName').addEventListener('input', (e) => {
        pairwiseLabelerName = e.target.value.trim();
    });
    
    // Pairwise notes input
    document.getElementById('pairwiseNotesInput').addEventListener('input', (e) => {
        pairwiseNotes = e.target.value;
    });
    
    // Pairwise comparison buttons
    document.querySelectorAll('.pairwise-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const recon = e.target.dataset.recon;
            const winner = e.target.dataset.winner;
            
            // Update button states for this reconstruction type
            document.querySelectorAll(`.pairwise-btn[data-recon="${recon}"]`).forEach(b => {
                b.classList.remove('active');
                b.style.backgroundColor = 'white';
                b.style.borderColor = '#ddd';
            });
            
            e.target.classList.add('active');
            e.target.style.backgroundColor = '#4a90e2';
            e.target.style.borderColor = '#4a90e2';
            e.target.style.color = 'white';
            
            pairwiseComparisons[recon] = winner;
        });
    });
    
    // Pairwise brightness sliders
    document.querySelectorAll('.brightness-slider-pairwise').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const imageNum = e.target.dataset.image;
            const value = parseInt(e.target.value);
            const valueSpan = document.querySelector(`.brightness-value-pairwise[data-image="${imageNum}"]`);
            const img = document.getElementById(`pairwiseImage${imageNum}`);
            
            if (valueSpan) valueSpan.textContent = `${value}%`;
            if (img) img.style.filter = `brightness(${value}%)`;
        });
    });
    
    // Pairwise save button
    document.getElementById('pairwiseSaveBtn').addEventListener('click', savePairwiseComparison);
    
    // Pairwise next button
    document.getElementById('pairwiseNextBtn').addEventListener('click', () => {
        savePairwiseComparison(true); // Save and load next
    });
    
    // Pairwise clear button
    document.getElementById('pairwiseClearBtn').addEventListener('click', clearPairwiseComparison);
}

// Load a random pair of images for comparison
function loadRandomPair() {
    if (images.length < 2) {
        alert('Need at least 2 images for pairwise comparison');
        return;
    }
    
    // Select two random different images
    let idx1 = Math.floor(Math.random() * images.length);
    let idx2 = Math.floor(Math.random() * images.length);
    while (idx2 === idx1) {
        idx2 = Math.floor(Math.random() * images.length);
    }
    
    pairwiseImage1 = images[idx1];
    pairwiseImage2 = images[idx2];
    
    // Load images
    const encodedPath1 = pairwiseImage1.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
    const encodedPath2 = pairwiseImage2.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
    
    const img1 = document.getElementById('pairwiseImage1');
    const img2 = document.getElementById('pairwiseImage2');
    
    img1.src = `/api/images/${encodedPath1}`;
    img2.src = `/api/images/${encodedPath2}`;
    
    img1.onload = function() {
        const slider = document.querySelector('.brightness-slider-pairwise[data-image="1"]');
        const brightness = slider ? parseInt(slider.value) : 100;
        this.style.filter = `brightness(${brightness}%)`;
    };
    
    img2.onload = function() {
        const slider = document.querySelector('.brightness-slider-pairwise[data-image="2"]');
        const brightness = slider ? parseInt(slider.value) : 100;
        this.style.filter = `brightness(${brightness}%)`;
    };
    
    document.getElementById('pairwiseImage1Info').textContent = pairwiseImage1.name;
    document.getElementById('pairwiseImage2Info').textContent = pairwiseImage2.name;
    
    // Clear previous comparisons
    clearPairwiseComparison();
}

// Save pairwise comparison
async function savePairwiseComparison(loadNext = false) {
    if (!pairwiseImage1 || !pairwiseImage2) {
        alert('Please wait for images to load');
        return;
    }
    
    if (!pairwiseLabelerName) {
        alert('Please enter your name before saving');
        return;
    }
    
    // Check if at least one comparison was made
    const comparisons = Object.keys(pairwiseComparisons);
    if (comparisons.length === 0) {
        alert('Please make at least one comparison before saving');
        return;
    }
    
    try {
        // Save each comparison separately
        const savePromises = comparisons.map(recon => {
            return fetch('/api/pairwise', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image1_path: pairwiseImage1.path,
                    image1_name: pairwiseImage1.name,
                    image2_path: pairwiseImage2.path,
                    image2_name: pairwiseImage2.name,
                    reconstruction_type: recon,
                    winner: pairwiseComparisons[recon],
                    labeler_name: pairwiseLabelerName,
                    notes: pairwiseNotes || null
                })
            });
        });
        
        const results = await Promise.all(savePromises);
        const allOk = results.every(r => r.ok);
        
        if (allOk) {
            if (loadNext) {
                loadRandomPair();
            } else {
                alert('Comparison saved successfully!');
            }
        } else {
            alert('Error saving some comparisons');
        }
    } catch (error) {
        console.error('Error saving pairwise comparison:', error);
        alert('Error saving comparison');
    }
}

// Clear pairwise comparison
function clearPairwiseComparison() {
    pairwiseComparisons = {};
    pairwiseNotes = '';
    document.getElementById('pairwiseNotesInput').value = '';
    
    // Reset all pairwise buttons
    document.querySelectorAll('.pairwise-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.style.backgroundColor = 'white';
        btn.style.borderColor = '#ddd';
        btn.style.color = '#333';
    });
    
    // Reset brightness sliders
    document.querySelectorAll('.brightness-slider-pairwise').forEach(slider => {
        slider.value = 100;
        const imageNum = slider.dataset.image;
        const valueSpan = document.querySelector(`.brightness-value-pairwise[data-image="${imageNum}"]`);
        const img = document.getElementById(`pairwiseImage${imageNum}`);
        if (valueSpan) valueSpan.textContent = '100%';
        if (img) img.style.filter = 'brightness(100%)';
    });
}

