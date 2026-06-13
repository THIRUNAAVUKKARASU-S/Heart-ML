// Heart Disease Prediction Web Application - Client Logic
document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = themeToggleBtn.querySelector("i");
    const navLinks = document.querySelectorAll(".nav-link");
    const sections = document.querySelectorAll(".content-section");
    const startPredictionBtn = document.getElementById("startPredictionBtn");
    
    // Form Inputs
    const predictionForm = document.getElementById("predictionForm");
    const ageInput = document.getElementById("age");
    const sexInput = document.getElementById("sex");
    const cpInput = document.getElementById("cp");
    const trestbpsInput = document.getElementById("trestbps");
    const cholInput = document.getElementById("chol");
    const modelInput = document.getElementById("model");
    const modelDesc = document.getElementById("modelDesc");
    
    // Result Card
    const resultCard = document.getElementById("resultCard");
    const resultStatus = document.getElementById("resultStatus");
    const resultConfidence = document.getElementById("resultConfidence");
    const riskMeter = document.getElementById("riskMeter");
    const loaderOverlay = document.getElementById("loaderOverlay");
    const exportReportBtn = document.getElementById("exportReportBtn");
    
    // Tables
    const historyTableBody = document.getElementById("historyTableBody");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    
    // Chart References
    let barChart = null;
    let radarChart = null;
    let doughnutChart = null;
    
    // Model descriptions
    const modelDescriptions = {
        svm: "Support Vector Machine with a linear kernel. It finds a high-dimensional hyperplane that maximizes the margin between risk groups, offering a robust baseline.",
        knn: "K-Nearest Neighbors using 5 nearest patient records for voting. It classifies risk based on similarity in age, sex, chest pain, blood pressure, and cholesterol.",
        gb: "Gradient Boosting Classifier. An ensemble technique that fits sequential decision trees to minimize training errors, achieving exceptional predictive accuracy.",
        jfo: "SVM with Jellyfish Optimization (JFO). Uses a custom metaheuristic algorithm to optimize the hyper-parameters (C, gamma) of an RBF kernel SVM, trained on Chi-Square selected features."
    };

    // --- Tab / Section Navigation ---
    function switchSection(targetSectionId) {
        sections.forEach(sec => {
            sec.classList.add("d-none");
            if (sec.id === targetSectionId) {
                sec.classList.remove("d-none");
                sec.classList.add("fade-in");
            }
        });
        
        navLinks.forEach(link => {
            if (link.getAttribute("data-target") === targetSectionId) {
                link.classList.add("active");
            } else {
                link.classList.remove("active");
            }
        });
        
        // If switching to Analytics, render charts
        if (targetSectionId === "analytics-section") {
            loadAndRenderMetrics();
        }
    }
    
    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const targetSection = link.getAttribute("data-target");
            switchSection(targetSection);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
    
    if (startPredictionBtn) {
        startPredictionBtn.addEventListener("click", (e) => {
            e.preventDefault();
            switchSection("prediction-section");
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // --- Theme Toggling ---
    const currentTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeIcon(currentTheme);
    
    themeToggleBtn.addEventListener("click", () => {
        let theme = document.documentElement.getAttribute("data-theme");
        if (theme === "dark") {
            theme = "light";
        } else {
            theme = "dark";
        }
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("theme", theme);
        updateThemeIcon(theme);
        
        // Re-render charts for color consistency
        if (!document.getElementById("analytics-section").classList.contains("d-none")) {
            loadAndRenderMetrics();
        }
    });
    
    function updateThemeIcon(theme) {
        if (theme === "dark") {
            themeIcon.className = "bi bi-sun-fill text-warning";
        } else {
            themeIcon.className = "bi bi-moon-fill text-primary";
        }
    }

    // --- Validation Logic ---
    function validateField(input, min, max, feedbackId) {
        const val = parseFloat(input.value);
        const feedback = document.getElementById(feedbackId);
        
        if (isNaN(val) || val < min || val > max) {
            input.classList.add("is-invalid-custom");
            feedback.style.display = "block";
            return false;
        } else {
            input.classList.remove("is-invalid-custom");
            feedback.style.display = "none";
            return true;
        }
    }
    
    function validateForm() {
        const isAgeValid = validateField(ageInput, 1, 120, "ageFeedback");
        const isBpsValid = validateField(trestbpsInput, 80, 220, "bpsFeedback");
        const isCholValid = validateField(cholInput, 100, 600, "cholFeedback");
        return isAgeValid && isBpsValid && isCholValid;
    }
    
    [ageInput, trestbpsInput, cholInput].forEach(input => {
        input.addEventListener("input", () => {
            validateForm();
        });
    });

    // --- Prediction Logic ---
    let debounceTimer = null;
    
    function triggerPrediction() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (validateForm()) {
                sendPredictionRequest();
            }
        }, 500);
    }
    
    // Event listeners for real-time predictions
    [ageInput, sexInput, cpInput, trestbpsInput, cholInput, modelInput].forEach(element => {
        element.addEventListener("change", () => {
            if (element === modelInput) {
                modelDesc.textContent = modelDescriptions[modelInput.value];
            }
            triggerPrediction();
        });
    });
    
    ageInput.addEventListener("input", triggerPrediction);
    trestbpsInput.addEventListener("input", triggerPrediction);
    cholInput.addEventListener("input", triggerPrediction);

    async function sendPredictionRequest() {
        loaderOverlay.classList.add("active");
        
        const payload = {
            age: parseInt(ageInput.value),
            sex: parseInt(sexInput.value),
            cp: parseInt(cpInput.value),
            trestbps: parseInt(trestbpsInput.value),
            chol: parseInt(cholInput.value),
            model: modelInput.value
        };
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error("Prediction API call failed");
            }
            
            const data = await response.json();
            displayPredictionResult(data, payload);
        } catch (error) {
            console.error("Error during prediction:", error);
            resultStatus.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill"></i> Error running model</span>`;
            resultConfidence.textContent = "--";
        } finally {
            loaderOverlay.classList.remove("active");
        }
    }
    
    function displayPredictionResult(data, inputs) {
        const hasDisease = data.prediction === "Heart Disease Detected";
        const confidence = data.confidence;
        
        // Status Text & Icon
        if (hasDisease) {
            resultStatus.innerHTML = `<i class="bi bi-heart-pulse-fill pulse-icon"></i> Heart Disease Detected`;
            resultStatus.className = "result-status status-disease";
        } else {
            resultStatus.innerHTML = `<i class="bi bi-check-circle-fill"></i> Low Risk`;
            resultStatus.className = "result-status status-healthy";
        }
        
        // Confidence
        resultConfidence.textContent = `${confidence.toFixed(1)}%`;
        
        // Risk Meter (Green <= 40%, Yellow 40-70%, Red > 70%)
        // The probability represents the likelihood of heart disease
        const probability = hasDisease ? confidence : (100 - confidence);
        riskMeter.style.width = `${probability}%`;
        
        riskMeter.className = "progress-bar risk-progress-bar";
        if (probability <= 40) {
            riskMeter.classList.add("risk-bg-low");
        } else if (probability <= 70) {
            riskMeter.classList.add("risk-bg-mod");
        } else {
            riskMeter.classList.add("risk-bg-high");
        }
        
        // Show Export Report button
        exportReportBtn.classList.remove("d-none");
        exportReportBtn.href = `/export?age=${inputs.age}&sex=${inputs.sex}&cp=${inputs.cp}&trestbps=${inputs.trestbps}&chol=${inputs.chol}&model=${inputs.model}&pred=${encodeURIComponent(data.prediction)}&conf=${confidence.toFixed(1)}`;
        
        // Save to History
        saveToHistory({
            date: new Date().toLocaleString(),
            inputs: inputs,
            prediction: data.prediction,
            confidence: confidence
        });
        
        // Update Doughnut Chart (Risk vs. Healthy)
        updateDoughnutChart(probability);
    }

    // --- History Logic ---
    function saveToHistory(entry) {
        let history = JSON.parse(localStorage.getItem("pred_history")) || [];
        // Prevent duplicate consecutive entries
        if (history.length > 0) {
            const last = history[0];
            if (JSON.stringify(last.inputs) === JSON.stringify(entry.inputs) && last.prediction === entry.prediction) {
                return;
            }
        }
        
        history.unshift(entry);
        // Limit history to last 10 entries
        if (history.length > 10) {
            history.pop();
        }
        localStorage.setItem("pred_history", JSON.stringify(history));
        renderHistory();
    }
    
    function renderHistory() {
        const history = JSON.parse(localStorage.getItem("pred_history")) || [];
        historyTableBody.innerHTML = "";
        
        if (history.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="6" class="text-center text-muted py-3">No predictions recorded yet.</td></tr>`;
            return;
        }
        
        const cpLabels = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"];
        
        history.forEach((entry, index) => {
            const sexLabel = entry.inputs.sex === 1 ? "Male" : "Female";
            const cpLabel = cpLabels[entry.inputs.cp] || "Unknown";
            const modelLabel = entry.inputs.model.toUpperCase();
            
            const isDisease = entry.prediction === "Heart Disease Detected";
            const badgeClass = isDisease ? "bg-danger" : "bg-success";
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${entry.date}</td>
                <td>Age: ${entry.inputs.age}, ${sexLabel}<br><small class="text-muted">BP: ${entry.inputs.trestbps}, Chol: ${entry.inputs.chol}, Chest Pain: ${cpLabel}</small></td>
                <td><span class="badge bg-secondary">${modelLabel}</span></td>
                <td><span class="badge ${badgeClass}">${entry.prediction}</span></td>
                <td><strong>${entry.confidence.toFixed(1)}%</strong></td>
                <td>
                    <a href="/export?age=${entry.inputs.age}&sex=${entry.inputs.sex}&cp=${entry.inputs.cp}&trestbps=${entry.inputs.trestbps}&chol=${entry.inputs.chol}&model=${entry.inputs.model}&pred=${encodeURIComponent(entry.prediction)}&conf=${entry.confidence.toFixed(1)}" class="btn btn-sm btn-outline-primary" title="Export PDF">
                        <i class="bi bi-file-earmark-pdf"></i>
                    </a>
                </td>
            `;
            historyTableBody.appendChild(tr);
        });
    }
    
    clearHistoryBtn.addEventListener("click", () => {
        localStorage.removeItem("pred_history");
        renderHistory();
    });

    // --- Charting & Metrics Logic ---
    async function loadAndRenderMetrics() {
        try {
            const response = await fetch("/metrics");
            if (!response.ok) {
                throw new Error("Failed to load metrics");
            }
            const data = await response.json();
            
            // Populate Metrics HTML Table
            updateMetricsTable(data);
            
            // Render Bar and Radar Charts
            renderBarChart(data);
            renderRadarChart(data);
        } catch (error) {
            console.error("Error loading metrics charts:", error);
        }
    }
    
    function updateMetricsTable(data) {
        const modelKeys = ["svm", "knn", "gb", "jfo"];
        modelKeys.forEach(key => {
            if (data[key]) {
                document.getElementById(`${key}-acc`).textContent = `${(data[key].accuracy * 100).toFixed(1)}%`;
                document.getElementById(`${key}-pre`).textContent = `${(data[key].precision * 100).toFixed(1)}%`;
                document.getElementById(`${key}-rec`).textContent = `${(data[key].recall * 100).toFixed(1)}%`;
                document.getElementById(`${key}-f1`).textContent = `${(data[key].f1 * 100).toFixed(1)}%`;
            }
        });
    }
    
    function getChartThemeColors() {
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        return {
            text: isDark ? "#f1f5f9" : "#0f172a",
            grid: isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.05)",
            colors: {
                svm: "rgba(37, 99, 235, 0.8)",    // Blue
                knn: "rgba(13, 148, 136, 0.8)",   // Teal
                gb: "rgba(245, 158, 11, 0.8)",    // Amber
                jfo: "rgba(168, 85, 247, 0.8)"     // Purple
            },
            borders: {
                svm: "rgb(37, 99, 235)",
                knn: "rgb(13, 148, 136)",
                gb: "rgb(245, 158, 11)",
                jfo: "rgb(168, 85, 247)"
            }
        };
    }
    
    function renderBarChart(metrics) {
        const ctx = document.getElementById("comparisonBarChart").getContext("2d");
        const theme = getChartThemeColors();
        
        const chartData = {
            labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
            datasets: [
                {
                    label: "SVM",
                    data: [metrics.svm.accuracy, metrics.svm.precision, metrics.svm.recall, metrics.svm.f1],
                    backgroundColor: theme.colors.svm,
                    borderColor: theme.borders.svm,
                    borderWidth: 1
                },
                {
                    label: "KNN",
                    data: [metrics.knn.accuracy, metrics.knn.precision, metrics.knn.recall, metrics.knn.f1],
                    backgroundColor: theme.colors.knn,
                    borderColor: theme.borders.knn,
                    borderWidth: 1
                },
                {
                    label: "Gradient Boosting",
                    data: [metrics.gb.accuracy, metrics.gb.precision, metrics.gb.recall, metrics.gb.f1],
                    backgroundColor: theme.colors.gb,
                    borderColor: theme.borders.gb,
                    borderWidth: 1
                },
                {
                    label: "SVM + JFO",
                    data: [metrics.jfo.accuracy, metrics.jfo.precision, metrics.jfo.recall, metrics.jfo.f1],
                    backgroundColor: theme.colors.jfo,
                    borderColor: theme.borders.jfo,
                    borderWidth: 1
                }
            ]
        };
        
        if (barChart) {
            barChart.destroy();
        }
        
        barChart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: theme.text, font: { family: 'Inter' } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.grid },
                        ticks: { color: theme.text }
                    },
                    y: {
                        min: 0.5,
                        max: 1.0,
                        grid: { color: theme.grid },
                        ticks: {
                            color: theme.text,
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    function renderRadarChart(metrics) {
        const ctx = document.getElementById("comparisonRadarChart").getContext("2d");
        const theme = getChartThemeColors();
        
        const chartData = {
            labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
            datasets: [
                {
                    label: "SVM",
                    data: [metrics.svm.accuracy, metrics.svm.precision, metrics.svm.recall, metrics.svm.f1],
                    backgroundColor: "rgba(37, 99, 235, 0.1)",
                    borderColor: theme.borders.svm,
                    pointBackgroundColor: theme.borders.svm
                },
                {
                    label: "KNN",
                    data: [metrics.knn.accuracy, metrics.knn.precision, metrics.knn.recall, metrics.knn.f1],
                    backgroundColor: "rgba(13, 148, 136, 0.1)",
                    borderColor: theme.borders.knn,
                    pointBackgroundColor: theme.borders.knn
                },
                {
                    label: "Gradient Boosting",
                    data: [metrics.gb.accuracy, metrics.gb.precision, metrics.gb.recall, metrics.gb.f1],
                    backgroundColor: "rgba(245, 158, 11, 0.1)",
                    borderColor: theme.borders.gb,
                    pointBackgroundColor: theme.borders.gb
                },
                {
                    label: "SVM + JFO",
                    data: [metrics.jfo.accuracy, metrics.jfo.precision, metrics.jfo.recall, metrics.jfo.f1],
                    backgroundColor: "rgba(168, 85, 247, 0.1)",
                    borderColor: theme.borders.jfo,
                    pointBackgroundColor: theme.borders.jfo
                }
            ]
        };
        
        if (radarChart) {
            radarChart.destroy();
        }
        
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: theme.text, font: { family: 'Inter' } }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: theme.grid },
                        grid: { color: theme.grid },
                        pointLabels: { color: theme.text, font: { family: 'Inter', size: 12 } },
                        ticks: {
                            color: theme.text,
                            backdropColor: 'transparent',
                            showLabelBackdrop: false
                        },
                        min: 0.5,
                        max: 1.0
                    }
                }
            }
        });
    }
    
    function updateDoughnutChart(probabilityVal) {
        const ctx = document.getElementById("confidenceDoughnutChart").getContext("2d");
        const theme = getChartThemeColors();
        
        // probabilityVal is risk of disease. Healthy is 100 - probabilityVal.
        const healthyVal = 100 - probabilityVal;
        
        const chartData = {
            labels: ["Disease Risk", "Healthy Probability"],
            datasets: [{
                data: [probabilityVal, healthyVal],
                backgroundColor: [
                    probabilityVal > 50 ? "rgba(239, 68, 68, 0.85)" : "rgba(245, 158, 11, 0.85)", // Red or Yellow
                    "rgba(16, 185, 129, 0.85)" // Green
                ],
                borderColor: [
                    probabilityVal > 50 ? "rgb(239, 68, 68)" : "rgb(245, 158, 11)",
                    "rgb(16, 185, 129)"
                ],
                borderWidth: 1
            }]
        };
        
        if (doughnutChart) {
            doughnutChart.destroy();
        }
        
        doughnutChart = new Chart(ctx, {
            type: 'doughnut',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: theme.text, font: { family: 'Inter' } }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // Initialize Page
    renderHistory();
    // Load prediction initial description
    modelDesc.textContent = modelDescriptions[modelInput.value];
    
    // Set initial doughnut chart (0% disease risk, 100% healthy, placeholder)
    updateDoughnutChart(0);
});
