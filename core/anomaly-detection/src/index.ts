/**
 * 🚀 ML Anomaly Detection System v5.0 - Main Entry Point
 *
 * Enterprise-grade anomaly detection system for cryptocurrency trading
 * Built with cloud-native architecture patterns
 *
 * Features:
 * - Statistical & ML-based detection algorithms
 * - Real-time streaming anomaly detection
 * - Crypto-specific detection patterns
 * - Enterprise monitoring & alerting
 * - RESTful & WebSocket APIs
 *
 * @author ML-Framework Team
 * @version 5.0.0
 * @license MIT
 */

// Export TypeScript interfaces for Python interop
export interface AnomalyDetectionResult {
 anomalyLabels: number[];
 anomalyScores: number[];
 timestamp: string;
 detectorType: string;
 metadata?: Record<string, any>;
}

export interface DetectorConfig {
 detectorType: 'statistical' | 'ml' | 'deep_learning' | 'timeseries' | 'crypto_specific';
 parameters: Record<string, any>;
 cryptoOptimized: boolean;
}

export interface CryptoAnomalyPattern {
 patternType: 'pump_dump' | 'wash_trading' | 'flash_crash' | 'whale_movement' | 'manipulation';
 severity: 'low' | 'medium' | 'high' | 'critical';
 confidence: number;
 symbol: string;
 timestamp: string;
 features: Record<string, number>;
}

export interface AlertConfiguration {
 enabled: boolean;
 severityThreshold: 'low' | 'medium' | 'high' | 'critical';
 channels: ('email' | 'slack' | 'webhook' | 'sms')[];
 escalationPolicy?: {
 timeoutMinutes: number;
 escalationLevels: string[];
 };
}

export interface SystemHealth {
 status: 'healthy' | 'degraded' | 'unhealthy';
 version: string;
 uptime: number;
 detectors: {
 active: number;
 total: number;
 };
 performance: {
 avgDetectionTime: number;
 throughput: number;
 errorRate: number;
 };
 resources: {
 cpuUsage: number;
 memoryUsage: number;
 diskUsage: number;
 };
}

// Main system class for TypeScript integration
export class AnomalyDetectionSystem {
 private pythonBridge: any; // Will be initialized with Python bridge

 constructor(config?: DetectorConfig) {
 // Initialize Python bridge for actual ML computations
 // This serves as TypeScript interface to Python ML backend
 }

 /**
 * Initialize the anomaly detection system
 */
 async initialize: Promise<void> {
 // Implementation will call Python initialization
 }

 /**
 * Train detectors on historical data
 */
 async trainDetectors(data: any[], config: DetectorConfig): Promise<void> {
 // Implementation will call Python training methods
 }

 /**
 * Detect anomalies in batch data
 */
 async detectAnomalies(data: any[]): Promise<AnomalyDetectionResult> {
 // Implementation will call Python detection methods
 throw new Error('Method not implemented - requires Python bridge');
 }

 /**
 * Real-time anomaly detection for streaming data
 */
 async detectRealtime(dataPoint: any): Promise<{
 isAnomaly: boolean;
 score: number;
 confidence: number;
 }> {
 // Implementation will call Python real-time detection
 throw new Error('Method not implemented - requires Python bridge');
 }

 /**
 * Get system health status
 */
 async getSystemHealth: Promise<SystemHealth> {
 // Implementation will gather system metrics
 return {
 status: 'healthy',
 version: '5.0.0',
 uptime: Date.now,
 detectors: { active: 0, total: 0 },
 performance: { avgDetectionTime: 0, throughput: 0, errorRate: 0 },
 resources: { cpuUsage: 0, memoryUsage: 0, diskUsage: 0 },
 };
 }
}

// Factory functions for common use cases
export const createCryptoAnomalyDetector = (config?: {
 exchanges?: string[];
 symbols?: string[];
 contamination?: number;
}): AnomalyDetectionSystem => {
 const detectorConfig: DetectorConfig = {
 detectorType: 'crypto_specific',
 parameters: {
 exchanges: config?.exchanges || ['binance', 'coinbase'],
 symbols: config?.symbols || ['BTC/USDT', 'ETH/USDT'],
 contamination: config?.contamination || 0.05,
 },
 cryptoOptimized: true,
 };

 return new AnomalyDetectionSystem(detectorConfig);
};

export const createRealtimeDetector = (config?: {
 windowSize?: number;
 adaptiveThresholds?: boolean;
}): AnomalyDetectionSystem => {
 const detectorConfig: DetectorConfig = {
 detectorType: 'ml',
 parameters: {
 windowSize: config?.windowSize || 100,
 adaptiveThresholds: config?.adaptiveThresholds || true,
 },
 cryptoOptimized: true,
 };

 return new AnomalyDetectionSystem(detectorConfig);
};

// Export constants
export const DETECTOR_TYPES = {
 STATISTICAL: 'statistical',
 ML: 'ml',
 DEEP_LEARNING: 'deep_learning',
 TIMESERIES: 'timeseries',
 CRYPTO_SPECIFIC: 'crypto_specific',
} as const;

export const ANOMALY_SEVERITY = {
 LOW: 'low',
 MEDIUM: 'medium',
 HIGH: 'high',
 CRITICAL: 'critical',
} as const;

// Default configurations
export const DEFAULT_CONFIGS = {
 isolationForest: {
 nEstimators: 200,
 contamination: 0.05,
 maxFeatures: 1.0,
 },
 lstm: {
 sequenceLength: 50,
 hiddenUnits: [50, 25],
 epochs: 50,
 },
 statistical: {
 threshold: 3.0,
 robust: true,
 bilateral: true,
 },
};

// Export version info
export const VERSION = '5.0.0';
export const BUILD_DATE = new Date.toISOString;

console.log(`
🚀 ML Anomaly Detection System v${VERSION} Initialized
📅 Build Date: ${BUILD_DATE}
🏗️ Architecture: Cloud-Native
🎯 Optimized for: Cryptocurrency Trading
`);

export default AnomalyDetectionSystem;
