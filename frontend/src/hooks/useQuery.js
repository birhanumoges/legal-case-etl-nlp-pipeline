import { useQuery as useRQ } from 'react-query'
import { analyticsAPI, casesAPI } from '../services/api'

export const useStats        = () => useRQ('stats',        analyticsAPI.stats,        { staleTime: 60000 })
export const useYearly       = () => useRQ('yearly',       analyticsAPI.yearly,       { staleTime: 60000 })
export const useForecast     = () => useRQ('forecast',     analyticsAPI.forecast,     { staleTime: 60000 })
export const useCaseTypeDist = () => useRQ('caseTypeDist', analyticsAPI.caseTypeDist, { staleTime: 60000 })
export const useVerdictDist  = () => useRQ('verdictDist',  analyticsAPI.verdictDist,  { staleTime: 60000 })
export const useSubTypeDist  = () => useRQ('subTypeDist',  analyticsAPI.subTypeDist,  { staleTime: 60000 })
export const useModels       = () => useRQ('models',       analyticsAPI.models,       { staleTime: 300000 })

export const useCaseList = (params) =>
  useRQ(['cases', params], () => casesAPI.list(params).then(r => r.data), { keepPreviousData: true })

export const useCaseDetail = (id) =>
  useRQ(['case', id], () => casesAPI.get(id).then(r => r.data), { enabled: !!id })
