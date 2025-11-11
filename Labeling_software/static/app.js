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
const imagesPerPage = 50; // Reduced from 100 to 50 for better performance

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
    loadPairwiseComparisons();
    setupEventListeners();
    initializePairwiseMode();
    setupLabelsTabs();
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
    document.getElementById('exportBtn').addEventListener('click', () => {
        // Export based on which tab is active
        const absoluteTab = document.getElementById('absoluteLabelsTab');
        if (absoluteTab.classList.contains('active')) {
            exportLabels();
        } else {
            exportPairwiseComparisons();
        }
    });
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

// Global state for pairwise comparisons
let pairwiseComparisonsList = [];

// Load pairwise comparisons from API
async function loadPairwiseComparisons() {
    try {
        const response = await fetch('/api/pairwise');
        pairwiseComparisonsList = await response.json();
        renderPairwiseTable();
    } catch (error) {
        console.error('Error loading pairwise comparisons:', error);
    }
}

// Render image grid
function renderImageGrid(filter = 'all') {
    const grid = document.getElementById('imageGrid');
    grid.innerHTML = '';
    
    // Disconnect previous observer if it exists (cleanup)
    if (window.imageObserver) {
        window.imageObserver.disconnect();
    }
    
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
        
        // Add loading placeholder
        const loadingPlaceholder = document.createElement('div');
        loadingPlaceholder.className = 'image-loading-placeholder';
        loadingPlaceholder.innerHTML = '<div class="loading-spinner"></div>';
        item.appendChild(loadingPlaceholder);
        
        const imgEl = document.createElement('img');
        // Properly encode the path - split by / and encode each segment
        const encodedPath = img.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
        const imageUrl = `/api/images/${encodedPath}`;
        imgEl.alt = img.name;
        imgEl.style.width = '100%';
        imgEl.style.height = '100%';
        imgEl.style.objectFit = 'contain';
        imgEl.style.display = 'none'; // Hide until loaded
        
        // Use data-src for lazy loading with Intersection Observer
        imgEl.dataset.src = imageUrl;
        
        imgEl.onerror = function() {
            console.error('Failed to load image:', img.path, 'URL:', imageUrl);
            loadingPlaceholder.innerHTML = '<div style="color: red; font-size: 10px;">Error</div>';
            this.style.display = 'none';
        };
        
        imgEl.onload = function() {
            // Hide placeholder and show image
            loadingPlaceholder.style.display = 'none';
            this.style.display = 'block';
        };
        
        const badge = document.createElement('div');
        badge.className = 'status-badge';
        const label = labels[img.path];
        if (label && label.reconstruction) {
            try {
                const recon = typeof label.reconstruction === 'string' ? JSON.parse(label.reconstruction) : label.reconstruction;
                if (Array.isArray(recon) && recon.length > 0) {
                    badge.textContent = '✓';
                    badge.classList.add('labeled');
                } else if (recon) {
                    badge.textContent = '✓';
                    badge.classList.add('labeled');
                }
            } catch {
                // Ignore parse errors
            }
        }
        
        item.appendChild(imgEl);
        if (badge.textContent) {
            item.appendChild(badge);
        }
        
        item.addEventListener('click', () => selectImage(img));
        grid.appendChild(item);
        
        // Use Intersection Observer for lazy loading
        if ('IntersectionObserver' in window) {
            // Create a single observer instance for all images (more efficient)
            if (!window.imageObserver) {
                window.imageObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            const img = entry.target;
                            if (img.dataset.src) {
                                img.src = img.dataset.src;
                                img.removeAttribute('data-src');
                            }
                            window.imageObserver.unobserve(img);
                        }
                    });
                }, {
                    root: null, // Use viewport as root
                    rootMargin: '200px', // Start loading 200px before image enters viewport
                    threshold: 0.01 // Trigger when 1% of image is visible
                });
            }
            
            // Observe the image element
            window.imageObserver.observe(imgEl);
        } else {
            // Fallback: load immediately if IntersectionObserver not supported
            imgEl.src = imageUrl;
        }
    });
    
    // Force load first few images immediately for better UX
    setTimeout(() => {
        const allImages = grid.querySelectorAll('.image-item img[data-src]');
        const imagesToLoadImmediately = Math.min(9, allImages.length); // Load first 9 (3x3 grid)
        
        // Load first 9 immediately
        for (let i = 0; i < imagesToLoadImmediately; i++) {
            if (allImages[i] && allImages[i].dataset.src) {
                allImages[i].src = allImages[i].dataset.src;
                allImages[i].removeAttribute('data-src');
                if (window.imageObserver) {
                    window.imageObserver.unobserve(allImages[i]);
                }
            }
        }
        
        // For remaining images, ensure they're all observed
        // Also check if any are already visible and should load immediately
        if (window.imageObserver) {
            const remainingImages = grid.querySelectorAll('.image-item img[data-src]');
            remainingImages.forEach(img => {
                // Check if image is already in viewport
                const rect = img.getBoundingClientRect();
                const isVisible = rect.top < window.innerHeight + 200 && rect.bottom > -200;
                
                if (isVisible && img.dataset.src) {
                    // Load immediately if already visible
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    window.imageObserver.unobserve(img);
                } else {
                    // Observe for lazy loading
                    window.imageObserver.observe(img);
                }
            });
        }
    }, 50); // Small delay to ensure DOM is ready
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
async function exportLabels() {
    try {
        const response = await fetch('/api/labels/export');
        const csv = await response.text();
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'labels_export.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting labels:', error);
        alert('Error exporting labels');
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
            // Reload pairwise comparisons to update table
            await loadPairwiseComparisons();
            
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

// Setup labels table tabs
function setupLabelsTabs() {
    document.getElementById('absoluteLabelsTab').addEventListener('click', () => {
        switchLabelsTab('absolute');
    });
    
    document.getElementById('pairwiseLabelsTab').addEventListener('click', () => {
        switchLabelsTab('pairwise');
    });
}

// Switch between absolute and pairwise labels tabs
function switchLabelsTab(tab) {
    // Update tab buttons
    document.getElementById('absoluteLabelsTab').classList.toggle('active', tab === 'absolute');
    document.getElementById('pairwiseLabelsTab').classList.toggle('active', tab === 'pairwise');
    
    // Show/hide sections
    document.getElementById('absoluteLabelsSection').style.display = tab === 'absolute' ? 'block' : 'none';
    document.getElementById('pairwiseLabelsSection').style.display = tab === 'pairwise' ? 'block' : 'none';
}

// Render pairwise comparisons table
function renderPairwiseTable() {
    const tbody = document.getElementById('pairwiseTableBody');
    tbody.innerHTML = '';
    
    if (pairwiseComparisonsList.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="7" style="text-align: center; padding: 20px; color: #999;">No pairwise comparisons yet</td>';
        tbody.appendChild(row);
        return;
    }
    
    pairwiseComparisonsList.forEach(comp => {
        const row = document.createElement('tr');
        
        let winnerText, winnerClass;
        if (comp.winner === '1') {
            winnerText = 'Image 1';
            winnerClass = 'winner-1';
        } else if (comp.winner === '2') {
            winnerText = 'Image 2';
            winnerClass = 'winner-2';
        } else if (comp.winner === 'tie') {
            winnerText = 'Tie';
            winnerClass = 'winner-tie';
        } else if (comp.winner === 'not_apply') {
            winnerText = 'Not Apply';
            winnerClass = 'winner-not-apply';
        } else {
            winnerText = comp.winner;
            winnerClass = '';
        }
        
        const notesText = comp.notes || '-';
        const notesDisplay = notesText.length > 30 ? notesText.substring(0, 30) + '...' : notesText;
        
        row.innerHTML = `
            <td>${comp.image1_name}</td>
            <td>${comp.image2_name}</td>
            <td>${comp.reconstruction_type}</td>
            <td class="${winnerClass}">${winnerText}</td>
            <td>${comp.labeler_name || '-'}</td>
            <td title="${notesText !== '-' ? notesText : ''}" style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${notesDisplay}</td>
            <td></td>
        `;
        
        // Add delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.className = 'delete-btn';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deletePairwiseComparison(comp.id, comp.image1_path, comp.image2_path, comp.reconstruction_type);
        };
        
        const actionCell = row.querySelector('td:last-child');
        actionCell.appendChild(deleteBtn);
        
        tbody.appendChild(row);
    });
}

// Delete a pairwise comparison
async function deletePairwiseComparison(compId, image1, image2, recon) {
    if (!confirm(`Delete comparison: ${image1.split('/').pop()} vs ${image2.split('/').pop()} (${recon})?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/pairwise/${compId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            // Reload pairwise comparisons
            await loadPairwiseComparisons();
            alert('Comparison deleted successfully!');
        } else {
            alert('Error deleting comparison: ' + (result.message || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting pairwise comparison:', error);
        alert('Error deleting comparison');
    }
}

