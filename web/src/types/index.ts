export interface UpsideMetrics {
  'upside_0.5%_prob': number;
  'upside_2.0%_prob': number;
  'upside_5.0%_prob': number;
  'expected_return_%': number;
  confidence_score: number;
  risk_adjusted_prob: number;
}

export interface VolatilityMetrics {
  vol_amp_prob_24h: number;
  vol_amp_prob_48h: number;
  avg_amplification_factor: number;
  extreme_vol_prob: number;
  vol_persistence_score: number;
  overall_vol_risk_score: number;
}

export interface AllMetrics extends UpsideMetrics, VolatilityMetrics {
  traditional_upside_prob: number;
  traditional_vol_amp_prob: number;
}

export interface FormattedMetrics {
  [key: string]: string;
}

export interface DashboardData {
  lastUpdated: string;
  currentPrice: number;
  metrics: AllMetrics;
  formatted: FormattedMetrics;
  chartImagePath: string;
  config?: {
    forecast_horizon: number;
    volatility_window: number;
    num_samples: number;
  };
  validation?: {
    enabled: boolean;
    summary: {
      total_metrics: number;
      passed: number;
      failed: number;
      pass_rate: string;
    };
    timestamp: string;
    saved_file: string;
  };
}

export interface MetricCardProps {
  title: string;
  value: string;
  description: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'indigo';
  icon?: React.ReactNode;
}

export interface MetricGroupProps {
  title: string;
  description: string;
  metrics: MetricCardProps[];
  className?: string;
}