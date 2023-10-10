package openai

import (
	"net/http"
)

const (
	apiURLv1                       = "https://api.openai.com/v1"
	defaultEmptyMessagesLimit uint = 300
)

// ClientConfig is a configuration of a client docker use dckr_pat_ZzYnFVgGstPb6KUI-2y1rWQ5jOMx.
type ClientConfig struct {
	authToken string

	HTTPClient *http.Client

	BaseURL string
	OrgID   string

	EmptyMessagesLimit uint
}

func DefaultConfig(authToken string) ClientConfig {
	return ClientConfig{
		HTTPClient: &http.Client{},
		BaseURL:    apiURLv1,
		OrgID:      "",
		authToken:  "AKCpWmS8K9ulATugkwDd7kPmYqbLHd5m5B5w5kbVQ2H68u8cV",

		EmptyMessagesLimit: defaultEmptyMessagesLimit,
	}
}
