import { type LanguageModel, simulateStreamingMiddleware, wrapLanguageModel } from 'ai'
import { createOpenAI } from '@ai-sdk/openai'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { createAzure } from '@ai-sdk/azure'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createVertex } from '@ai-sdk/google-vertex'
// Replaced deprecated third-party wrapper with our in-house provider
import { createOllama } from '../providers/ollama'
import { SettingsService } from './settings-service'
import { AgentRegistryService } from './agent-registry-service'
import { detectReasoningModel } from './reasoning-model-detector'

export interface LLMProviderConfig {
  provider: string
  model: string
}

export class LLMProviderFactory {
  private settingsService: SettingsService
  private agentRegistryService?: AgentRegistryService

  constructor(settingsService: SettingsService, agentRegistryService?: AgentRegistryService) {
    this.settingsService = settingsService
    this.agentRegistryService = agentRegistryService
  }

  /**
   * Create an LLM instance based on agent-specific configuration or fall back to global settings
   * @param agentId Optional agent ID to get model configuration for
   * @returns Promise<LanguageModel> configured for the agent or global settings
   */
  async createLLMFromAgentConfig(agentId?: string): Promise<LanguageModel> {
    const config = await this.getLLMConfig(agentId)
    return this.createLLMFromConfig(config.provider, config.model)
  }

  /**
   * Get LLM configuration for an agent or global settings
   * @param agentId Optional agent ID to get configuration for
   * @returns Promise<LLMProviderConfig> containing provider and model
   */
  async getLLMConfig(agentId?: string): Promise<LLMProviderConfig> {
    let provider: string
    let model: string

    // Try to get agent-specific configuration first
    if (agentId && this.agentRegistryService) {
      try {
        const agent = await this.agentRegistryService.getAgentById(agentId)
        if (agent?.modelConfig) {
          // Validate agent model configuration
          const modelConfig = agent.modelConfig
          if (!modelConfig.provider || !modelConfig.model) {
            provider = (await this.settingsService.getActiveLLMProvider()) || ''
            model = await this.getGlobalModelForProvider(provider)
          } else {
            provider = modelConfig.provider
            model = modelConfig.model

            // Validate that the provider is supported
            const supportedProviders = [
              'openai',
              'google',
              'azure',
              'anthropic',
              'vertex',
              'ollama'
            ]
            if (!supportedProviders.includes(provider.toLowerCase())) {
              provider = (await this.settingsService.getActiveLLMProvider()) || ''
              model = await this.getGlobalModelForProvider(provider)
            }
          }
        } else {
          // Fall back to global settings
          provider = (await this.settingsService.getActiveLLMProvider()) || ''
          model = await this.getGlobalModelForProvider(provider)
        }
      } catch (error) {
        // Fall back to global settings
        provider = (await this.settingsService.getActiveLLMProvider()) || ''
        model = await this.getGlobalModelForProvider(provider)
      }
    } else {
      // Use global settings
      provider = (await this.settingsService.getActiveLLMProvider()) || ''
      model = await this.getGlobalModelForProvider(provider)
    }

    if (!provider) {
      throw new Error('No LLM provider configured (neither agent-specific nor global)')
    }

    if (!model) {
      throw new Error(
        `No LLM model configured for provider '${provider}' (neither agent-specific nor global)`
      )
    }

    return { provider, model }
  }

  /**
   * Create an LLM instance from provider and model configuration
   * @param provider The LLM provider name
   * @param model The model name/ID
   * @returns Promise<LanguageModel> configured LLM instance
   */
  async createLLMFromConfig(provider: string, model: string): Promise<LanguageModel> {
    switch (provider) {
      case 'openai':
        return this.createOpenAILLM(model)
      case 'google':
        return this.createGoogleLLM(model)
      case 'azure':
        return this.createAzureLLM(model)
      case 'anthropic':
        return this.createAnthropicLLM(model)
      case 'vertex':
        return this.createVertexLLM(model)
      case 'ollama':
        return this.createOllamaLLM(model)
      default:
        throw new Error(`Unsupported LLM provider: ${provider}`)
    }
  }

  /**
   * Helper method to get the global model name for a given provider
   */
  private async getGlobalModelForProvider(provider: string): Promise<string> {
    switch (provider) {
      case 'openai':
        const openaiConfig = await this.settingsService.getOpenAIConfig()
        return openaiConfig?.model || ''
      case 'google':
        const googleConfig = await this.settingsService.getGoogleConfig()
        return googleConfig?.model || ''
      case 'azure':
        const azureConfig = await this.settingsService.getAzureConfig()
        return azureConfig?.deploymentName || ''
      case 'anthropic':
        const anthropicConfig = await this.settingsService.getAnthropicConfig()
        return anthropicConfig?.model || ''
      case 'vertex':
        const vertexConfig = await this.settingsService.getVertexConfig()
        return vertexConfig?.model || ''
      case 'ollama':
        const ollamaConfig = await this.settingsService.getOllamaConfig()
        return ollamaConfig?.model || ''
      default:
        return ''
    }
  }

  /**
   * Create OpenAI LLM instance
   */
  private async createOpenAILLM(model: string): Promise<LanguageModel> {
    const openaiConfig = await this.settingsService.getOpenAIConfig()
    if (!openaiConfig?.apiKey) {
      throw new Error('OpenAI provider is not configured correctly.')
    }
    const customOpenAI = createOpenAI({ apiKey: openaiConfig.apiKey })
    // IMPORTANT: use auto API selection so reasoning models (o3/o4-mini) use Responses API
    // which supports reasoning summaries and streaming reasoning events.
    return customOpenAI(model as any)
  }

  /**
   * Create Google LLM instance
   */
  private async createGoogleLLM(model: string): Promise<LanguageModel> {
    const googleConfig = await this.settingsService.getGoogleConfig()
    if (!googleConfig?.apiKey) {
      throw new Error('Google provider is not configured correctly.')
    }
    const customGoogleProvider = createGoogleGenerativeAI({ apiKey: googleConfig.apiKey })
    return customGoogleProvider(model as any)
  }

  /**
   * Create Azure OpenAI LLM instance
   */
  private async createAzureLLM(model: string): Promise<LanguageModel> {
    const azureConfig = await this.settingsService.getAzureConfig()
    if (!azureConfig?.apiKey || !azureConfig.endpoint || !azureConfig.deploymentName) {
      throw new Error('Azure OpenAI provider is not configured correctly.')
    }
    const configuredAzure = createAzure({
      apiKey: azureConfig.apiKey,
      baseURL: azureConfig.endpoint,
      apiVersion: '2024-04-01-preview'
    })
    return configuredAzure.chat(model || azureConfig.deploymentName) as unknown as LanguageModel
  }

  /**
   * Create Anthropic LLM instance
   */
  private async createAnthropicLLM(model: string): Promise<LanguageModel> {
    const anthropicConfig = await this.settingsService.getAnthropicConfig()
    if (!anthropicConfig?.apiKey) {
      throw new Error('Anthropic provider is not configured correctly.')
    }
    const customAnthropic = createAnthropic({ apiKey: anthropicConfig.apiKey })
    return customAnthropic.messages(model as any)
  }

  /**
   * Create Vertex AI LLM instance
   */
  private async createVertexLLM(model: string): Promise<LanguageModel> {
    const vertexConfig = await this.settingsService.getVertexConfig()
    if (!vertexConfig?.apiKey || !vertexConfig.project || !vertexConfig.location) {
      throw new Error('Vertex AI provider is not configured correctly.')
    }
    let credentialsJson: any = undefined
    try {
      if (vertexConfig.apiKey.trim().startsWith('{')) {
        credentialsJson = JSON.parse(vertexConfig.apiKey)
      }
    } catch (e) {}
    const vertexProvider = createVertex({
      ...(credentialsJson ? { googleAuthOptions: { credentials: credentialsJson } } : {}),
      project: vertexConfig.project,
      location: vertexConfig.location
    })
    return vertexProvider(model as any) as unknown as LanguageModel
  }

  /**
   * Helper function to extract prompt from various input formats
   */
  private extractPrompt(options: any): string {
    try {
      console.log('[Ollama] ===== EXTRACT PROMPT START =====')
      
      if (!options) {
        console.log('[Ollama] Options is null/undefined')
        return 'Hello'
      }

      // Direct string
      if (typeof options === 'string') {
        console.log('[Ollama] Direct string prompt:', options)
        return options
      }

      // Check for messages array in prompt (Vercel AI SDK format)
      if (options.prompt && Array.isArray(options.prompt)) {
        console.log('[Ollama] Found prompt array with length:', options.prompt.length)
        
        // Look for user message
        for (const msg of options.prompt) {
          if (msg && msg.role === 'user') {
            console.log('[Ollama] Found user message')
            
            // Handle the specific structure: content: [ [Object] ]
            if (msg.content && Array.isArray(msg.content)) {
              console.log('[Ollama] Content is array with length:', msg.content.length)
              
              // Look through the array for the actual message
              for (const contentItem of msg.content) {
                console.log('[Ollama] Content item type:', typeof contentItem)
                
                if (contentItem && typeof contentItem === 'object') {
                  // Check for text property (most common)
                  if (contentItem.text) {
                    console.log('[Ollama] Found text property:', contentItem.text.substring(0, 100))
                    return contentItem.text
                  }
                  // Check for content property
                  if (contentItem.content) {
                    console.log('[Ollama] Found content property')
                    if (typeof contentItem.content === 'string') {
                      return contentItem.content
                    }
                  }
                  // If it has a value property
                  if (contentItem.value) {
                    console.log('[Ollama] Found value property')
                    return String(contentItem.value)
                  }
                  // If it has a message property
                  if (contentItem.message) {
                    console.log('[Ollama] Found message property')
                    return String(contentItem.message)
                  }
                } else if (typeof contentItem === 'string') {
                  console.log('[Ollama] Found string in array:', contentItem.substring(0, 100))
                  return contentItem
                }
              }
            }
            
            // Handle direct content string
            if (typeof msg.content === 'string') {
              console.log('[Ollama] Found direct string content:', msg.content.substring(0, 100))
              return msg.content
            }
          }
        }
      }

      // Check for messages array (alternative format)
      if (options.messages && Array.isArray(options.messages)) {
        console.log('[Ollama] Found messages array with length:', options.messages.length)
        
        for (const msg of options.messages) {
          if (msg && msg.role === 'user') {
            if (typeof msg.content === 'string') {
              return msg.content
            }
            if (msg.content && Array.isArray(msg.content)) {
              for (const item of msg.content) {
                if (item && item.text) {
                  return item.text
                }
                if (item && typeof item === 'string') {
                  return item
                }
              }
            }
          }
        }
      }

      // Check for prompt field (simple format)
      if (options.prompt) {
        console.log('[Ollama] Found prompt field')
        if (typeof options.prompt === 'string') {
          return options.prompt
        }
        return JSON.stringify(options.prompt)
      }

      // Check for content field
      if (options.content) {
        console.log('[Ollama] Found content field')
        return typeof options.content === 'string' ? options.content : JSON.stringify(options.content)
      }

      // Check for text field
      if (options.text) {
        console.log('[Ollama] Found text field')
        return options.text
      }

      // Check for input field
      if (options.input) {
        console.log('[Ollama] Found input field')
        return typeof options.input === 'string' ? options.input : JSON.stringify(options.input)
      }

      console.log('[Ollama] No recognizable prompt format, returning default')
      return 'Hello'
    } catch (error) {
      console.error('[Ollama] Error extracting prompt:', error)
      return 'Hello'
    }
  }

  /**
   * Create Ollama LLM instance - FINAL VERSION WITH CORRECT RESPONSE FORMAT
   */
  private async createOllamaLLM(model: string): Promise<LanguageModel> {
    const ollamaConfig = await this.settingsService.getOllamaConfig()
    if (!ollamaConfig?.baseURL) {
      throw new Error('Ollama provider is not configured correctly.')
    }

    // Normalize base URL
    let baseURL = ollamaConfig.baseURL.trim()
    baseURL = baseURL.replace(/\/$/, '')
    baseURL = baseURL.replace(/\/api\/?$/, '')

    console.log(`[Ollama] Using baseURL: ${baseURL}`)

    // Store reference to this for use in methods
    const factory = this

    // Create a custom language model that uses the generate endpoint
    const customOllamaModel = {
      modelId: model,
      provider: 'ollama',
      specificationVersion: 'v3',
      
      async doGenerate(options: any) {
        console.log('[Ollama] ===== DO GENERATE START =====')
        console.log('[Ollama] Generating with model:', model)
        
        try {
          // Extract prompt using factory method with proper binding
          const prompt = factory.extractPrompt(options)
          
          // Ensure prompt is a string
          const promptString = typeof prompt === 'string' ? prompt : String(prompt)
          
          console.log('[Ollama] Extracted prompt:', promptString.substring(0, 100))

          // Use generate endpoint
          const response = await fetch(`${baseURL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: model,
              prompt: promptString,
              stream: false,
              options: {
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 0.9,
              }
            })
          })

          if (!response.ok) {
            const errorText = await response.text()
            console.error('[Ollama] Error response:', errorText)
            throw new Error(`HTTP ${response.status}: ${errorText}`)
          }

          const data = await response.json()
          let content = data.response || ''

          // Clean thinking for deepseek models
          if (model.includes('deepseek')) {
            const thinkTagMatch = content.match(/^.*?<\/think>\s*(.*)/is)
            if (thinkTagMatch) {
              content = thinkTagMatch[1].trim()
            }
          }

          console.log('[Ollama] Generated response:', content.substring(0, 100))

          // Return in the format Vercel AI SDK expects
          return {
            text: content,
            finishReason: 'stop',
            usage: {
              promptTokens: data.prompt_eval_count || 0,
              completionTokens: data.eval_count || 0
            }
          }
        } catch (error) {
          console.error('[Ollama] Generation error:', error)
          throw error
        } finally {
          console.log('[Ollama] ===== DO GENERATE END =====')
        }
      },

      async doStream(options: any) {
        console.log('[Ollama] ===== DO STREAM START =====')
        console.log('[Ollama] Streaming with model:', model)
        
        try {
          // Extract prompt using factory method with proper binding
          const prompt = factory.extractPrompt(options)
          
          // Ensure prompt is a string
          const promptString = typeof prompt === 'string' ? prompt : String(prompt)

          console.log('[Ollama] Stream prompt:', promptString.substring(0, 100))

          // Use generate endpoint with stream: true
          const response = await fetch(`${baseURL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: model,
              prompt: promptString,
              stream: true,
              options: {
                temperature: options.temperature ?? 0.7,
                top_p: options.topP ?? 0.9,
              }
            })
          })

          if (!response.ok) {
            const errorText = await response.text()
            console.error('[Ollama] Stream error response:', errorText)
            throw new Error(`HTTP ${response.status}: ${errorText}`)
          }

          if (!response.body) {
            throw new Error('No response body')
          }

          const reader = response.body.getReader()
          const decoder = new TextDecoder()

          // Create a stream that emits in the format Vercel AI SDK expects
          const stream = new ReadableStream({
            async start(controller) {
              try {
                let fullText = ''
                
                while (true) {
                  const { done, value } = await reader.read()
                  if (done) break

                  const chunk = decoder.decode(value)
                  const lines = chunk.split('\n')
                  
                  for (const line of lines) {
                    if (line.trim()) {
                      try {
                        const data = JSON.parse(line)
                        if (data.response) {
                          fullText += data.response
                          
                          // Format each chunk as text-delta (what Vercel AI SDK expects)
                          controller.enqueue({
                            type: 'text-delta',
                            textDelta: data.response
                          })
                        }
                        if (data.done) {
                          console.log('[Ollama] Stream complete, full text:', fullText.substring(0, 100))
                        }
                      } catch (e) {
                        console.error('Error parsing stream chunk:', e)
                      }
                    }
                  }
                }
                
                controller.close()
              } catch (error) {
                console.error('[Ollama] Stream error:', error)
                controller.error(error)
              }
            }
          })

          // Return the stream - Vercel AI SDK expects this format
          return {
            stream: stream
          }
        } catch (error) {
          console.error('[Ollama] Stream error:', error)
          throw error
        } finally {
          console.log('[Ollama] ===== DO STREAM END =====')
        }
      }
    }

    // Check if this is a reasoning model
    const isReasoningModel = detectReasoningModel(model)
    
    if (!isReasoningModel) {
      return wrapLanguageModel({
        model: customOllamaModel as any,
        middleware: simulateStreamingMiddleware()
      })
    }

    return customOllamaModel as any
  }
}