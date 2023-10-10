package openai

import (
	"context"
	"net/http"
)

type ModerationRequest struct {
	Input string  `json:"input,omitempty"`
	Model *string `json:"model,omitempty"`
}

type Result struct {
	Categories     ResultCategories     `json:"categories"`
	CategoryScores ResultCategoryScores `json:"category_scores"`
	Flagged        bool                 `json:"flagged"`
}

type ResultCategories struct {
	Hate            bool `json:"hate"`
	HateThreatening bool `json:"hate/threatening"`
	SelfHarm        bool `json:"self-harm"`
	Sexual          bool `json:"sexual"`
	SexualMinors    bool `json:"sexual/minors"`
	Violence        bool `json:"violence"`
	ViolenceGraphic bool `json:"violence/graphic"`
}

type ResultCategoryScores struct {
	Hate            float32 `json:"hate"`
	HateThreatening float32 `json:"hate/threatening"`
	SelfHarm        float32 `json:"self-harm"`
	Sexual          float32 `json:"sexual"`
	SexualMinors    float32 `json:"sexual/minors"`
	Violence        float32 `json:"violence"`
	ViolenceGraphic float32 `json:"violence/graphic"`
}

type ModerationResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Results []Result `json:"results"`
}

func (c *Client) Moderations(ctx context.Context, request ModerationRequest) (response ModerationResponse, err error) {
	req, err := c.requestBuilder.build(ctx, http.MethodPost, c.fullURL("/moderations"), request)
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}
