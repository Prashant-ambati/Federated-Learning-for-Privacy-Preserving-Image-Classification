// Mobile Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
    hamburger.classList.remove('active');
    navMenu.classList.remove('active');
}));

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar background on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Add fade-in class to elements and observe them
document.addEventListener('DOMContentLoaded', () => {
    const elementsToAnimate = document.querySelectorAll('.feature-card, .performance-card, .deployment-card, .tech-category, .detail-card');
    
    elementsToAnimate.forEach(el => {
        el.classList.add('fade-in');
        observer.observe(el);
    });
});

// Performance chart animation
const animateCharts = () => {
    const charts = document.querySelectorAll('.chart-bar');
    charts.forEach((chart, index) => {
        setTimeout(() => {
            const height = chart.style.height;
            chart.style.height = '0%';
            setTimeout(() => {
                chart.style.height = height;
            }, 100);
        }, index * 200);
    });
};

// Trigger chart animation when performance section is visible
const performanceSection = document.querySelector('.performance');
if (performanceSection) {
    const performanceObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCharts();
                performanceObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });
    
    performanceObserver.observe(performanceSection);
}

// Network diagram animation
const animateNetworkDiagram = () => {
    const lines = document.querySelectorAll('.connection-lines line');
    lines.forEach((line, index) => {
        line.style.strokeDasharray = '1000';
        line.style.strokeDashoffset = '1000';
        line.style.animation = `drawLine 2s ease-in-out ${index * 0.2}s forwards`;
    });
};

// Add CSS animation for line drawing
const style = document.createElement('style');
style.textContent = `
    @keyframes drawLine {
        to {
            stroke-dashoffset: 0;
        }
    }
`;
document.head.appendChild(style);

// Trigger network animation when hero section is visible
const heroSection = document.querySelector('.hero');
if (heroSection) {
    const heroObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setTimeout(animateNetworkDiagram, 1000);
                heroObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    heroObserver.observe(heroSection);
}

// Typing effect for hero title
const typeWriter = (element, text, speed = 100) => {
    let i = 0;
    element.innerHTML = '';
    
    const timer = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
        } else {
            clearInterval(timer);
        }
    }, speed);
};

// Initialize typing effect
document.addEventListener('DOMContentLoaded', () => {
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        setTimeout(() => {
            typeWriter(heroTitle, originalText, 50);
        }, 500);
    }
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroVisual = document.querySelector('.hero-visual');
    
    if (heroVisual) {
        const rate = scrolled * -0.5;
        heroVisual.style.transform = `translateY(${rate}px)`;
    }
});

// Copy to clipboard functionality for code snippets
document.querySelectorAll('.code-snippet').forEach(snippet => {
    snippet.addEventListener('click', () => {
        const text = snippet.textContent;
        navigator.clipboard.writeText(text).then(() => {
            // Show feedback
            const originalText = snippet.innerHTML;
            snippet.innerHTML = 'Copied!';
            snippet.style.background = '#10b981';
            
            setTimeout(() => {
                snippet.innerHTML = originalText;
                snippet.style.background = '';
            }, 1000);
        });
    });
    
    // Add cursor pointer
    snippet.style.cursor = 'pointer';
    snippet.title = 'Click to copy';
});

// Statistics counter animation
const animateCounters = () => {
    const counters = document.querySelectorAll('.stat-number');
    
    counters.forEach(counter => {
        const target = counter.textContent;
        const numericTarget = parseFloat(target.replace(/[^\d.]/g, ''));
        const suffix = target.replace(/[\d.]/g, '');
        
        let current = 0;
        const increment = numericTarget / 50;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= numericTarget) {
                counter.textContent = target;
                clearInterval(timer);
            } else {
                counter.textContent = Math.floor(current) + suffix;
            }
        }, 30);
    });
};

// Trigger counter animation when stats are visible
const statsSection = document.querySelector('.hero-stats');
if (statsSection) {
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounters();
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.8 });
    
    statsObserver.observe(statsSection);
}

// Add hover effects to cards
document.querySelectorAll('.feature-card, .deployment-card, .tech-category').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0) scale(1)';
    });
});

// Lazy loading for images (if any are added later)
const lazyImages = document.querySelectorAll('img[data-src]');
const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            imageObserver.unobserve(img);
        }
    });
});

lazyImages.forEach(img => imageObserver.observe(img));

// Add loading states
const addLoadingStates = () => {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', (e) => {
            if (button.href && button.href.includes('github.com')) {
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
                setTimeout(() => {
                    button.innerHTML = button.dataset.originalText || button.innerHTML;
                }, 2000);
            }
        });
        
        // Store original text
        button.dataset.originalText = button.innerHTML;
    });
};

// Initialize loading states
document.addEventListener('DOMContentLoaded', addLoadingStates);

// Performance monitoring
const performanceMetrics = {
    startTime: performance.now(),
    
    logMetric: function(name, value) {
        console.log(`Performance Metric - ${name}: ${value}ms`);
    },
    
    measurePageLoad: function() {
        window.addEventListener('load', () => {
            const loadTime = performance.now() - this.startTime;
            this.logMetric('Page Load Time', loadTime.toFixed(2));
        });
    }
};

performanceMetrics.measurePageLoad();

// Error handling
window.addEventListener('error', (e) => {
    console.error('JavaScript Error:', e.error);
});

// Service Worker registration (for future PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}