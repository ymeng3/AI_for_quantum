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

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadImages();
    loadLabels();
    setupEventListeners();
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
    window.location.href = '/api/labels/export';
}

