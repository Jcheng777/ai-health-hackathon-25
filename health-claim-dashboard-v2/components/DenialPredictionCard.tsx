'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Brain, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react'

interface ClaimData {
  procedureCode: string
  diagnosisCode: string
  insuranceType: string
  providerId?: string
  billedAmount: number
  allowedAmount?: number
  paidAmount?: number
  reasonCode?: string
  followUpRequired?: string
}

interface PredictionResult {
  prediction: 'approved' | 'denied' | 'review'
  confidence: number
  reasoning: string[]
  riskFactors: string[]
  error?: string
}

interface DenialPredictionCardProps {
  claimData: ClaimData
  onPredictionUpdate?: (prediction: PredictionResult) => void
}

export function DenialPredictionCard({ claimData, onPredictionUpdate }: DenialPredictionCardProps) {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handlePredict = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/predict-denial', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(claimData),
      })

      if (!response.ok) {
        throw new Error('Failed to get prediction')
      }

      const result: PredictionResult = await response.json()
      setPrediction(result)
      onPredictionUpdate?.(result)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  const getPredictionIcon = (pred: string) => {
    switch (pred) {
      case 'approved':
        return <CheckCircle className="h-6 w-6 text-green-500" />
      case 'denied':
        return <XCircle className="h-6 w-6 text-red-500" />
      case 'review':
        return <AlertTriangle className="h-6 w-6 text-yellow-500" />
      default:
        return <Brain className="h-6 w-6 text-gray-500" />
    }
  }

  const getPredictionColor = (pred: string) => {
    switch (pred) {
      case 'approved':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'denied':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'review':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          AI Denial Prediction
        </CardTitle>
        <CardDescription>
          Get AI-powered predictions for claim approval likelihood
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Claim Summary */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <p className="text-sm font-medium text-gray-600">Procedure Code</p>
            <p className="text-lg font-semibold">{claimData.procedureCode}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Diagnosis Code</p>
            <p className="text-lg font-semibold">{claimData.diagnosisCode}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Insurance Type</p>
            <p className="text-lg font-semibold">{claimData.insuranceType}</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-600">Billed Amount</p>
            <p className="text-lg font-semibold">${claimData.billedAmount.toLocaleString()}</p>
          </div>
        </div>

        {/* Prediction Button */}
        <Button 
          onClick={handlePredict} 
          disabled={isLoading}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing Claim...
            </>
          ) : (
            <>
              <Brain className="mr-2 h-4 w-4" />
              Get AI Prediction
            </>
          )}
        </Button>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="space-y-4">
            {/* Main Prediction */}
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center gap-3">
                {getPredictionIcon(prediction.prediction)}
                <div>
                  <p className="font-semibold text-lg capitalize">{prediction.prediction}</p>
                  <p className="text-sm text-gray-600">
                    Confidence: {' '}
                    <span className={`font-medium ${getConfidenceColor(prediction.confidence)}`}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </p>
                </div>
              </div>
              <Badge className={getPredictionColor(prediction.prediction)}>
                {prediction.prediction.toUpperCase()}
              </Badge>
            </div>

            {/* Confidence Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Prediction Confidence</span>
                <span className={getConfidenceColor(prediction.confidence)}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <Progress 
                value={prediction.confidence * 100} 
                className="h-2"
              />
            </div>

            {/* Reasoning */}
            {prediction.reasoning.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold text-sm">AI Reasoning:</h4>
                <ul className="space-y-1">
                  {prediction.reasoning.map((reason, index) => (
                    <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                      <span className="w-2 h-2 bg-blue-500 rounded-full mt-1.5 flex-shrink-0"></span>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risk Factors */}
            {prediction.riskFactors.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-semibold text-sm">Risk Factors:</h4>
                <div className="flex flex-wrap gap-2">
                  {prediction.riskFactors.map((factor, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {factor}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="font-semibold text-sm mb-2">Recommendations:</h4>
              <div className="text-sm text-blue-800">
                {prediction.prediction === 'approved' && (
                  <p>✓ Claim likely to be approved. Proceed with standard processing.</p>
                )}
                {prediction.prediction === 'denied' && (
                  <p>⚠ High risk of denial. Consider reviewing documentation and addressing risk factors before submission.</p>
                )}
                {prediction.prediction === 'review' && (
                  <p>📋 Manual review recommended. Uncertain prediction requires human evaluation.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 