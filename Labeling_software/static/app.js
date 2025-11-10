// Global state
let images = [];
let labels = {};
let currentImage = null;
let currentLabels = {
    quality: null,
    reconstruction: null
};
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
    
    // Label buttons
    document.querySelectorAll('.label-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const type = e.target.dataset.type;
            const value = e.target.dataset.value;
            
            // Toggle active state
            document.querySelectorAll(`.label-btn[data-type="${type}"]`).forEach(b => {
                b.classList.remove('active');
            });
            e.target.classList.add('active');
            
            currentLabels[type] = value;
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
        
        if (label && label.quality && label.reconstruction) {
            if (!badge) {
                badge = document.createElement('div');
                badge.className = 'status-badge';
                item.appendChild(badge);
            }
            badge.textContent = '✓';
            badge.className = 'status-badge labeled';
        } else if (label && (label.quality || label.reconstruction)) {
            if (!badge) {
                badge = document.createElement('div');
                badge.className = 'status-badge';
                item.appendChild(badge);
            }
            badge.textContent = '~';
            badge.className = 'status-badge partial';
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
        
        currentLabels.quality = labelData.quality || null;
        currentLabels.reconstruction = labelData.reconstruction || null;
        
        // Update button states
        updateButtonStates();
    } catch (error) {
        console.error('Error loading label:', error);
        currentLabels.quality = null;
        currentLabels.reconstruction = null;
        updateButtonStates();
    }
}

// Update button states based on current labels
function updateButtonStates() {
    // Clear all active states
    document.querySelectorAll('.label-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Set active states
    if (currentLabels.quality) {
        const btn = document.querySelector(`.label-btn[data-type="quality"][data-value="${currentLabels.quality}"]`);
        if (btn) btn.classList.add('active');
    }
    
    if (currentLabels.reconstruction) {
        const btn = document.querySelector(`.label-btn[data-type="reconstruction"][data-value="${currentLabels.reconstruction}"]`);
        if (btn) btn.classList.add('active');
    }
}

// Save label
async function saveLabel() {
    if (!currentImage) {
        alert('Please select an image first');
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
                quality: currentLabels.quality,
                reconstruction: currentLabels.reconstruction
            })
        });
        
        if (response.ok) {
            await loadLabels(); // Reload labels to update table and grid
            alert('Label saved successfully!');
        } else {
            alert('Error saving label');
        }
    } catch (error) {
        console.error('Error saving label:', error);
        alert('Error saving label');
    }
}

// Clear labels
function clearLabels() {
    currentLabels.quality = null;
    currentLabels.reconstruction = null;
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
        
        row.innerHTML = `
            <td>${label.file_name}</td>
            <td>${label.quality || '-'}</td>
            <td>${label.reconstruction || '-'}</td>
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

